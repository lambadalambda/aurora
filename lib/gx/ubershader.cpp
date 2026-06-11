#include "ubershader.hpp"

#include "../gfx/common.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "shader_info.hpp"

#include <bit>
#include <mutex>

namespace aurora::gx::uber {
static Module Log("aurora::gx::uber");

using webgpu::g_device;

// ---------------------------------------------------------------------------
// Interpreter encoding (must match the WGSL below):
//
// attr word (one u32 per GX attr slot, 21 slots):
//   [0:2)  attrType: 0 none, 1 direct, 2 index8, 3 index16
//   [2:5)  component count (1-4)
//   [5:9)  component type: 0 u8, 1 s8, 2 u16, 3 s16, 4 f32,
//          5 rgb565, 6 rgb8, 7 rgbx8, 8 rgba4, 9 rgba6, 10 rgba8
//   [9:14) frac
//   [14]   little-endian array data
//   [16:24) offset within vertex
//   [24:32) array stride
//
// tcg word (one u32 per texgen, 8 used):
//   [0:2)  type: 0 mtx2x4, 1 mtx3x4
//   [2:6)  source: 0 pos, 1 nrm, 2+k = tex{k}
//   [6:11) static palette slot (mtx/3), 31 = identity
//   [11]   matrix index comes from the per-vertex texmtxidx attribute
//   [12]   normalize
//   [13]   entry valid (referenced by a stage and supported)
//
// channel word (4: color0, color1, alpha0, alpha1):
//   [0] lightingEnabled  [1] ambSrc==VTX  [2] matSrc==VTX
//   [3:5) diffFn: 0 none, 1 sign, 2 clamp
//   [5:7) attnFn: 0 none, 1 spot, 2 spec
//
// stage words (vec4u per stage):
//   x: color args a,b,c,d in [0:4)[4:8)[8:12)[12:16) (GXTevColorArg values),
//      alpha args a,b,c,d in [16:19)[19:22)[22:25)[25:28) (GXTevAlphaArg)
//   y: color op[0:4) bias[4:6) scale[6:8) clamp[8] outReg[9:11),
//      alpha op[16:20) bias[20:22) scale[22:24) clamp[24] outReg[25:27)
//   z: kcSel[0:5) kaSel[5:10) chan[10:12) (0/1 = rast index, 2 = zero)
//      texCoord[12:16) texMap[16:20) (15 = unused) texSwap[20:22) rasSwap[22:24)
//
// swap_tables: 4 tables x 8 bits (r[0:2) g[2:4) b[4:6) a[6:8)).
// tex_uv_map.x: 4 bits per texture map = texcoord index used to sample it.
// fog_type: 0 lin, 1 exp, 2 exp2, 3 revexp, 4 revexp2, 7 off.
// alpha_comp: comp0[0:3) comp1[3:6) op[6:8) ref0[8:16) ref1[16:24),
//             0xFFFFFFFF = always pass.
// ---------------------------------------------------------------------------

constexpr char UberShaderSource[] = R"""(
struct Light {
    pos: vec3f,
    dir: vec3f,
    color: vec4f,
    cos_att: vec3f,
    dist_att: vec3f,
};
struct Fog {
    color: vec4f,
    a: f32,
    b: f32,
    c: f32,
    pad: f32,
};
struct UberUniform {
    vtx_start: u32,
    current_pnmtx: u32,
    vtx_stride: u32,
    num_stages: u32,
    num_texgens: u32,
    fog_type: u32,
    alpha_comp: u32,
    swap_tables: u32,
    tex_uv_map: vec4u,
    array_start: array<vec4u, 3>,
    mtx_start: array<vec4u, 8>,
    attrs: array<vec4u, 6>,
    tcgs: array<vec4u, 2>,
    chans: vec4u,
    stages: array<vec4u, 16>,
    proj: mat4x4f,
    tevregs: array<vec4f, 4>,
    kcolor: array<vec4f, 4>,
    chan_amb: array<vec4f, 4>,
    chan_mat: array<vec4f, 4>,
    light_state: vec4u,
    lights: array<Light, 8>,
    fog: Fog,
    tex_size_bias: array<vec4f, 8>,
};
@group(0) @binding(0) var<storage, read> vbuf: array<u32>;
@group(0) @binding(1) var<storage, read> abuf: array<u32>;
@group(1) @binding(0) var<uniform> ubuf: UberUniform;
@group(2) @binding(0) var tex0: texture_2d<f32>;
@group(2) @binding(1) var tex0_samp: sampler;
@group(2) @binding(2) var tex1: texture_2d<f32>;
@group(2) @binding(3) var tex1_samp: sampler;
@group(2) @binding(4) var tex2: texture_2d<f32>;
@group(2) @binding(5) var tex2_samp: sampler;
@group(2) @binding(6) var tex3: texture_2d<f32>;
@group(2) @binding(7) var tex3_samp: sampler;
@group(2) @binding(8) var tex4: texture_2d<f32>;
@group(2) @binding(9) var tex4_samp: sampler;
@group(2) @binding(10) var tex5: texture_2d<f32>;
@group(2) @binding(11) var tex5_samp: sampler;
@group(2) @binding(12) var tex6: texture_2d<f32>;
@group(2) @binding(13) var tex6_samp: sampler;
@group(2) @binding(14) var tex7: texture_2d<f32>;
@group(2) @binding(15) var tex7_samp: sampler;

// --- raw byte loads (vertex stream is big-endian unless flagged) ---
fn ld_u8(p: ptr<storage, array<u32>, read>, off: u32) -> u32 {
    return ((*p)[off >> 2u] >> ((off & 3u) * 8u)) & 0xffu;
}
fn ld_u16(p: ptr<storage, array<u32>, read>, off: u32, le: bool) -> u32 {
    let b0 = ld_u8(p, off);
    let b1 = ld_u8(p, off + 1u);
    return select((b0 << 8u) | b1, (b1 << 8u) | b0, le);
}
fn ld_u32(p: ptr<storage, array<u32>, read>, off: u32, le: bool) -> u32 {
    let h0 = ld_u16(p, off, le);
    let h1 = ld_u16(p, off + 2u, le);
    return select((h0 << 16u) | h1, (h1 << 16u) | h0, le);
}
fn ld_comp(p: ptr<storage, array<u32>, read>, off: u32, ctype: u32, frac: u32, le: bool) -> f32 {
    switch ctype {
        case 0u: { return f32(ld_u8(p, off)) / f32(1u << frac); }
        case 1u: { return f32(bitcast<i32>(ld_u8(p, off) << 24u) >> 24u) / f32(1u << frac); }
        case 2u: { return f32(ld_u16(p, off, le)) / f32(1u << frac); }
        case 3u: { return f32(bitcast<i32>(ld_u16(p, off, le) << 16u) >> 16u) / f32(1u << frac); }
        default: { return bitcast<f32>(ld_u32(p, off, le)); }
    }
}
fn comp_size(ctype: u32) -> u32 {
    switch ctype {
        case 0u, 1u: { return 1u; }
        case 2u, 3u: { return 2u; }
        default: { return 4u; }
    }
}
fn ld_color(p: ptr<storage, array<u32>, read>, off: u32, ctype: u32, le: bool) -> vec4f {
    switch ctype {
        case 5u: { // rgb565
            let v = ld_u16(p, off, le);
            return vec4f(f32(v >> 11u) / 31.0, f32((v >> 5u) & 0x3fu) / 63.0, f32(v & 0x1fu) / 31.0, 1.0);
        }
        case 6u: { // rgb8
            return vec4f(f32(ld_u8(p, off)) / 255.0, f32(ld_u8(p, off + 1u)) / 255.0,
                         f32(ld_u8(p, off + 2u)) / 255.0, 1.0);
        }
        case 7u: { // rgbx8
            return vec4f(f32(ld_u8(p, off)) / 255.0, f32(ld_u8(p, off + 1u)) / 255.0,
                         f32(ld_u8(p, off + 2u)) / 255.0, 1.0);
        }
        case 8u: { // rgba4
            let v = ld_u16(p, off, le);
            return vec4f(f32(v >> 12u) / 15.0, f32((v >> 8u) & 0xfu) / 15.0, f32((v >> 4u) & 0xfu) / 15.0,
                         f32(v & 0xfu) / 15.0);
        }
        case 9u: { // rgba6
            let v = (ld_u8(p, off) << 16u) | (ld_u8(p, off + 1u) << 8u) | ld_u8(p, off + 2u);
            return vec4f(f32(v >> 18u) / 63.0, f32((v >> 12u) & 0x3fu) / 63.0, f32((v >> 6u) & 0x3fu) / 63.0,
                         f32(v & 0x3fu) / 63.0);
        }
        default: { // rgba8
            return vec4f(f32(ld_u8(p, off)) / 255.0, f32(ld_u8(p, off + 1u)) / 255.0,
                         f32(ld_u8(p, off + 2u)) / 255.0, f32(ld_u8(p, off + 3u)) / 255.0);
        }
    }
}

// --- attribute decode ---
fn attr_word(a: u32) -> u32 { return ubuf.attrs[a >> 2u][a & 3u]; }
fn array_base(a: u32) -> u32 {
    let i = a - 9u; // GX_VA_POS
    return ubuf.array_start[i >> 2u][i & 3u];
}
fn mtx_off(slot: u32) -> u32 { return ubuf.mtx_start[slot >> 2u][slot & 3u]; }
fn fetch_mtx34(offset: u32) -> mat3x4f {
    let i = offset / 4u;
    return mat3x4f(
        vec4f(bitcast<f32>(abuf[i]), bitcast<f32>(abuf[i + 1u]), bitcast<f32>(abuf[i + 2u]),
              bitcast<f32>(abuf[i + 3u])),
        vec4f(bitcast<f32>(abuf[i + 4u]), bitcast<f32>(abuf[i + 5u]), bitcast<f32>(abuf[i + 6u]),
              bitcast<f32>(abuf[i + 7u])),
        vec4f(bitcast<f32>(abuf[i + 8u]), bitcast<f32>(abuf[i + 9u]), bitcast<f32>(abuf[i + 10u]),
              bitcast<f32>(abuf[i + 11u])));
}

// Resolves the byte address of an attribute's data and which buffer it lives
// in. Returns vec2u(address, in_abuf).
fn attr_addr(a: u32, w: u32, vidx: u32) -> vec2u {
    let ty = w & 3u;
    let dl_off = ubuf.vtx_start + vidx * ubuf.vtx_stride + ((w >> 16u) & 0xffu);
    if (ty == 1u) { // direct
        return vec2u(dl_off, 0u);
    }
    var index: u32;
    if (ty == 2u) {
        index = ld_u8(&vbuf, dl_off);
    } else {
        index = ld_u16(&vbuf, dl_off, false);
    }
    return vec2u(array_base(a) + index * ((w >> 24u) & 0xffu), 1u);
}
fn fetch_attr(a: u32, vidx: u32, def: vec4f) -> vec4f {
    let w = attr_word(a);
    if ((w & 3u) == 0u) {
        return def;
    }
    let addr = attr_addr(a, w, vidx);
    let cnt = (w >> 2u) & 7u;
    let ctype = (w >> 5u) & 0xfu;
    let frac = (w >> 9u) & 0x1fu;
    let le = ((w >> 14u) & 1u) != 0u;
    if (ctype >= 5u) { // color format
        if (addr.y != 0u) {
            return ld_color(&abuf, addr.x, ctype, le);
        }
        return ld_color(&vbuf, addr.x, ctype, le);
    }
    var out = def;
    let sz = comp_size(ctype);
    for (var c = 0u; c < cnt; c++) {
        if (addr.y != 0u) {
            out[c] = ld_comp(&abuf, addr.x + c * sz, ctype, frac, le);
        } else {
            out[c] = ld_comp(&vbuf, addr.x + c * sz, ctype, frac, le);
        }
    }
    return out;
}
// Matrix index attributes are direct u8 palette offsets (multiples of 3).
fn fetch_mtxidx(a: u32, vidx: u32, def: u32) -> u32 {
    let w = attr_word(a);
    if ((w & 3u) == 0u) {
        return def;
    }
    let dl_off = ubuf.vtx_start + vidx * ubuf.vtx_stride + ((w >> 16u) & 0xffu);
    return ld_u8(&vbuf, dl_off) / 3u;
}

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) mv_pos: vec3f,
    @location(1) mv_nrm: vec3f,
    @location(2) clr0: vec4f,
    @location(3) clr1: vec4f,
    @location(4) uvw0: vec3f,
    @location(5) uvw1: vec3f,
    @location(6) uvw2: vec3f,
    @location(7) uvw3: vec3f,
    @location(8) uvw4: vec3f,
    @location(9) uvw5: vec3f,
    @location(10) uvw6: vec3f,
    @location(11) uvw7: vec3f,
};

@vertex
fn vs_main(@builtin(vertex_index) vidx: u32) -> VertexOutput {
    var out: VertexOutput;
    let pn_slot = fetch_mtxidx(0u, vidx, ubuf.current_pnmtx);
    let in_pos = fetch_attr(9u, vidx, vec4f(0.0)).xyz;
    let mv_pos = vec4f(in_pos, 1.0) * fetch_mtx34(mtx_off(pn_slot));
    out.pos = vec4f(mv_pos, 1.0) * ubuf.proj;
    out.pos.z = -out.pos.z; // reversed Z (mirrors UseReversedZ in shader.cpp)
    let in_nrm = fetch_attr(10u, vidx, vec4f(1.0, 0.0, 0.0, 0.0)).xyz;
    let nrm_tmp = vec4f(in_nrm, 0.0) * fetch_mtx34(mtx_off(20u + pn_slot));
    let mv_nrm = select(nrm_tmp, normalize(nrm_tmp), dot(nrm_tmp, nrm_tmp) > 1e-10);
    out.mv_pos = mv_pos;
    out.mv_nrm = mv_nrm;
    out.clr0 = fetch_attr(11u, vidx, vec4f(0.0));
    out.clr1 = fetch_attr(12u, vidx, vec4f(0.0));

    var uvw: array<vec3f, 8>;
    for (var i = 0u; i < 8u; i++) {
        uvw[i] = vec3f(0.0, 0.0, 1.0);
    }
    for (var i = 0u; i < 8u; i++) {
        let w = ubuf.tcgs[i >> 2u][i & 3u];
        if (((w >> 13u) & 1u) == 0u) {
            continue;
        }
        let src = (w >> 2u) & 0xfu;
        var tc: vec4f;
        if (src == 0u) {
            tc = vec4f(in_pos, 1.0);
        } else if (src == 1u) {
            tc = vec4f(in_nrm, 1.0);
        } else {
            let uv = fetch_attr(13u + (src - 2u), vidx, vec4f(0.0));
            tc = vec4f(uv.xy, 1.0, 1.0);
        }
        var slot = (w >> 6u) & 0x1fu;
        if (((w >> 11u) & 1u) != 0u) {
            slot = fetch_mtxidx(1u + i, vidx, slot);
        }
        var tmp: vec3f;
        if (slot == 31u) {
            tmp = tc.xyz;
        } else {
            tmp = tc * fetch_mtx34(mtx_off(slot));
        }
        if ((w & 3u) == 0u) { // mtx2x4
            tmp.z = 1.0;
        }
        if (((w >> 12u) & 1u) != 0u) {
            tmp = normalize(tmp);
        }
        uvw[i] = tmp;
    }
    out.uvw0 = uvw[0];
    out.uvw1 = uvw[1];
    out.uvw2 = uvw[2];
    out.uvw3 = uvw[3];
    out.uvw4 = uvw[4];
    out.uvw5 = uvw[5];
    out.uvw6 = uvw[6];
    out.uvw7 = uvw[7];
    return out;
}

// --- TEV interpreter state ---
var<private> regs: array<vec4f, 4>;
var<private> rast: array<vec4f, 2>;
var<private> samp_tex: array<vec4f, 8>;

fn tev_overflow3(v: vec3f) -> vec3f {
    let b = v * 255.0;
    return (b - floor(b / 256.0) * 256.0) / 255.0;
}
fn tev_overflow1(v: f32) -> f32 {
    let b = v * 255.0;
    return (b - floor(b / 256.0) * 256.0) / 255.0;
}
fn swap4(v: vec4f, sw: u32) -> vec4f {
    return vec4f(v[sw & 3u], v[(sw >> 2u) & 3u], v[(sw >> 4u) & 3u], v[(sw >> 6u) & 3u]);
}
fn konst_c(sel: u32) -> vec3f {
    if (sel < 8u) {
        return vec3f(f32(8u - sel) / 8.0);
    }
    if (sel < 12u) {
        return vec3f(1.0);
    }
    if (sel < 16u) {
        return ubuf.kcolor[sel - 12u].rgb;
    }
    let reg = (sel - 16u) & 3u;
    let comp = (sel - 16u) >> 2u;
    return vec3f(ubuf.kcolor[reg][comp]);
}
fn konst_a(sel: u32) -> f32 {
    if (sel < 8u) {
        return f32(8u - sel) / 8.0;
    }
    if (sel < 16u) {
        return 1.0;
    }
    let reg = (sel - 16u) & 3u;
    let comp = (sel - 16u) >> 2u;
    return ubuf.kcolor[reg][comp];
}
fn color_arg(arg: u32, texc: vec4f, rasc: vec4f, kc: vec3f) -> vec3f {
    switch arg {
        case 0u: { return regs[0].rgb; }
        case 1u: { return vec3f(regs[0].a); }
        case 2u: { return regs[1].rgb; }
        case 3u: { return vec3f(regs[1].a); }
        case 4u: { return regs[2].rgb; }
        case 5u: { return vec3f(regs[2].a); }
        case 6u: { return regs[3].rgb; }
        case 7u: { return vec3f(regs[3].a); }
        case 8u: { return texc.rgb; }
        case 9u: { return vec3f(texc.a); }
        case 10u: { return rasc.rgb; }
        case 11u: { return vec3f(rasc.a); }
        case 12u: { return vec3f(1.0); }
        case 13u: { return vec3f(0.5); }
        case 14u: { return kc; }
        default: { return vec3f(0.0); }
    }
}
fn alpha_arg(arg: u32, texa: f32, rasa: f32, ka: f32) -> f32 {
    switch arg {
        case 0u: { return regs[0].a; }
        case 1u: { return regs[1].a; }
        case 2u: { return regs[2].a; }
        case 3u: { return regs[3].a; }
        case 4u: { return texa; }
        case 5u: { return rasa; }
        case 6u: { return ka; }
        default: { return 0.0; }
    }
}
fn tev_op3(op: u32, a: vec3f, b: vec3f, c: vec3f, d: vec3f, bias: u32, scale: u32) -> vec3f {
    let oa = tev_overflow3(a);
    let ob = tev_overflow3(b);
    if (op < 2u) {
        let biasv = vec3f(select(select(0.0, 0.5, bias == 1u), -0.5, bias == 2u));
        let scalev = select(select(select(1.0, 2.0, scale == 1u), 4.0, scale == 2u), 0.5, scale == 3u);
        let m = mix(oa, ob, c);
        return (select(m, -m, op == 1u) + d + biasv) * scalev;
    }
    switch op {
        case 8u: { return select(vec3f(0.0), c, round(oa.r * 255.0) > round(ob.r * 255.0)) + d; }
        case 9u: { return select(vec3f(0.0), c, round(oa.r * 255.0) == round(ob.r * 255.0)) + d; }
        case 10u: {
            return select(vec3f(0.0), c,
                          round(dot(oa.rg * 255.0, vec2f(1.0, 256.0))) > round(dot(ob.rg * 255.0, vec2f(1.0, 256.0)))) +
                   d;
        }
        case 11u: {
            return select(vec3f(0.0), c,
                          round(dot(oa.rg * 255.0, vec2f(1.0, 256.0))) ==
                              round(dot(ob.rg * 255.0, vec2f(1.0, 256.0)))) +
                   d;
        }
        case 12u: {
            return select(vec3f(0.0), c,
                          round(dot(oa * 255.0, vec3f(1.0, 256.0, 65536.0))) >
                              round(dot(ob * 255.0, vec3f(1.0, 256.0, 65536.0)))) +
                   d;
        }
        case 13u: {
            return select(vec3f(0.0), c,
                          round(dot(oa * 255.0, vec3f(1.0, 256.0, 65536.0))) ==
                              round(dot(ob * 255.0, vec3f(1.0, 256.0, 65536.0)))) +
                   d;
        }
        case 14u: {
            return select(vec3f(0.0), c, round(oa * 255.0) > round(ob * 255.0)) + d;
        }
        default: {
            return select(vec3f(0.0), c, round(oa * 255.0) == round(ob * 255.0)) + d;
        }
    }
}
fn tev_op1(op: u32, a: f32, b: f32, c: f32, d: f32, bias: u32, scale: u32) -> f32 {
    let oa = tev_overflow1(a);
    let ob = tev_overflow1(b);
    if (op < 2u) {
        let biasv = select(select(0.0, 0.5, bias == 1u), -0.5, bias == 2u);
        let scalev = select(select(select(1.0, 2.0, scale == 1u), 4.0, scale == 2u), 0.5, scale == 3u);
        let m = mix(oa, ob, c);
        return (select(m, -m, op == 1u) + d + biasv) * scalev;
    }
    switch op {
        case 8u: { return select(0.0, c, round(oa * 255.0) > round(ob * 255.0)) + d; }
        default: { return select(0.0, c, round(oa * 255.0) == round(ob * 255.0)) + d; }
    }
}

fn light_sum(amb: vec4f, mask: u32, diff_fn: u32, attn_fn: u32, pos: vec3f, nrm: vec3f) -> vec4f {
    var lighting = amb;
    for (var i = 0u; i < 8u; i++) {
        if ((mask & (1u << i)) == 0u) {
            continue;
        }
        let light = ubuf.lights[i];
        var ldir = light.pos - pos;
        let dist2 = dot(ldir, ldir);
        let dist = sqrt(dist2);
        ldir = ldir / dist;
        var attn = 1.0;
        if (attn_fn == 1u) { // spot
            let cosine = max(0.0, dot(ldir, light.dir));
            let cos_attn = dot(light.cos_att, vec3f(1.0, cosine, cosine * cosine));
            let dist_attn = dot(light.dist_att, vec3f(1.0, dist, dist2));
            attn = max(0.0, cos_attn / dist_attn);
        } else if (attn_fn == 2u) { // spec
            attn = select(0.0, max(0.0, dot(nrm, light.dir)), dot(nrm, ldir) >= 0.0);
            let cos_attn = dot(light.cos_att, vec3f(1.0, attn, attn * attn));
            var dist_attn: f32;
            if (diff_fn != 0u) {
                dist_attn = max(0.0, dot(normalize(light.dist_att), vec3f(1.0, attn, attn * attn)));
            } else {
                dist_attn = max(0.0, dot(light.dist_att, vec3f(1.0, attn, attn * attn)));
            }
            attn = max(0.0, cos_attn / dist_attn);
        }
        var diff = 1.0;
        if (diff_fn == 1u) {
            diff = dot(ldir, nrm);
        } else if (diff_fn == 2u) {
            diff = max(0.0, dot(ldir, nrm));
        }
        lighting = lighting + (attn * diff * light.color);
    }
    return lighting;
}

fn compute_rast(ch: u32, clr: vec4f, pos: vec3f, nrm: vec3f) {
    let wc = ubuf.chans[ch];
    let wa = ubuf.chans[2u + ch];
    let matc = select(ubuf.chan_mat[ch], clr, ((wc >> 2u) & 1u) != 0u);
    var color = matc;
    if ((wc & 1u) != 0u) {
        let ambc = select(ubuf.chan_amb[ch], clr, ((wc >> 1u) & 1u) != 0u);
        color = matc * clamp(light_sum(ambc, ubuf.light_state[ch], (wc >> 3u) & 3u, (wc >> 5u) & 3u, pos, nrm),
                             vec4f(0.0), vec4f(1.0));
    }
    let mata = select(ubuf.chan_mat[2u + ch], clr, ((wa >> 2u) & 1u) != 0u);
    var alpha = mata;
    if ((wa & 1u) != 0u) {
        let amba = select(ubuf.chan_amb[2u + ch], clr, ((wa >> 1u) & 1u) != 0u);
        alpha = mata * clamp(light_sum(amba, ubuf.light_state[2u + ch], (wa >> 3u) & 3u, (wa >> 5u) & 3u, pos, nrm),
                             vec4f(0.0), vec4f(1.0));
    }
    rast[ch] = vec4f(color.rgb, alpha.a);
}

fn alpha_cmp(a: u32, comp: u32, ref_val: u32) -> bool {
    switch comp {
        case 0u: { return false; }
        case 1u: { return a < ref_val; }
        case 2u: { return a == ref_val; }
        case 3u: { return a <= ref_val; }
        case 4u: { return a > ref_val; }
        case 5u: { return a != ref_val; }
        case 6u: { return a >= ref_val; }
        default: { return true; }
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var uv: array<vec2f, 8>;
    uv[0] = in.uvw0.xy / in.uvw0.z;
    uv[1] = in.uvw1.xy / in.uvw1.z;
    uv[2] = in.uvw2.xy / in.uvw2.z;
    uv[3] = in.uvw3.xy / in.uvw3.z;
    uv[4] = in.uvw4.xy / in.uvw4.z;
    uv[5] = in.uvw5.xy / in.uvw5.z;
    uv[6] = in.uvw6.xy / in.uvw6.z;
    uv[7] = in.uvw7.xy / in.uvw7.z;
    let m = ubuf.tex_uv_map.x;
    samp_tex[0] = textureSampleBias(tex0, tex0_samp, uv[m & 7u], ubuf.tex_size_bias[0].z);
    samp_tex[1] = textureSampleBias(tex1, tex1_samp, uv[(m >> 4u) & 7u], ubuf.tex_size_bias[1].z);
    samp_tex[2] = textureSampleBias(tex2, tex2_samp, uv[(m >> 8u) & 7u], ubuf.tex_size_bias[2].z);
    samp_tex[3] = textureSampleBias(tex3, tex3_samp, uv[(m >> 12u) & 7u], ubuf.tex_size_bias[3].z);
    samp_tex[4] = textureSampleBias(tex4, tex4_samp, uv[(m >> 16u) & 7u], ubuf.tex_size_bias[4].z);
    samp_tex[5] = textureSampleBias(tex5, tex5_samp, uv[(m >> 20u) & 7u], ubuf.tex_size_bias[5].z);
    samp_tex[6] = textureSampleBias(tex6, tex6_samp, uv[(m >> 24u) & 7u], ubuf.tex_size_bias[6].z);
    samp_tex[7] = textureSampleBias(tex7, tex7_samp, uv[(m >> 28u) & 7u], ubuf.tex_size_bias[7].z);

    compute_rast(0u, in.clr0, in.mv_pos, in.mv_nrm);
    compute_rast(1u, in.clr1, in.mv_pos, in.mv_nrm);

    regs[0] = ubuf.tevregs[0];
    regs[1] = ubuf.tevregs[1];
    regs[2] = ubuf.tevregs[2];
    regs[3] = ubuf.tevregs[3];

    for (var s = 0u; s < ubuf.num_stages; s++) {
        let w0 = ubuf.stages[s].x;
        let w1 = ubuf.stages[s].y;
        let w2 = ubuf.stages[s].z;
        let tex_swap = (ubuf.swap_tables >> (((w2 >> 20u) & 3u) * 8u)) & 0xffu;
        let ras_swap = (ubuf.swap_tables >> (((w2 >> 22u) & 3u) * 8u)) & 0xffu;
        let tex_map = (w2 >> 16u) & 0xfu;
        let chan = (w2 >> 10u) & 3u;
        let texc = swap4(samp_tex[tex_map & 7u], tex_swap);
        let rasc = select(vec4f(0.0), swap4(rast[chan & 1u], ras_swap), chan < 2u);
        let kc = konst_c(w2 & 0x1fu);
        let ka = konst_a((w2 >> 5u) & 0x1fu);

        let ca = color_arg(w0 & 0xfu, texc, rasc, kc);
        let cb = color_arg((w0 >> 4u) & 0xfu, texc, rasc, kc);
        let cc = color_arg((w0 >> 8u) & 0xfu, texc, rasc, kc);
        let cd = color_arg((w0 >> 12u) & 0xfu, texc, rasc, kc);
        let cop = w1 & 0xfu;
        var cres = tev_op3(cop, ca, cb, cc, cd, (w1 >> 4u) & 3u, (w1 >> 6u) & 3u);
        if (((w1 >> 8u) & 1u) != 0u) {
            cres = clamp(cres, vec3f(0.0), vec3f(1.0));
        } else {
            cres = clamp(cres, vec3f(-4.0), vec3f(4.0));
        }
        let coutreg = (w1 >> 9u) & 3u;
        regs[coutreg] = vec4f(cres, regs[coutreg].a);

        let aa = alpha_arg((w0 >> 16u) & 7u, texc.a, rast[chan & 1u][(ras_swap >> 6u) & 3u], ka);
        let ab = alpha_arg((w0 >> 19u) & 7u, texc.a, rast[chan & 1u][(ras_swap >> 6u) & 3u], ka);
        let ac = alpha_arg((w0 >> 22u) & 7u, texc.a, rast[chan & 1u][(ras_swap >> 6u) & 3u], ka);
        let ad = alpha_arg((w0 >> 25u) & 7u, texc.a, rast[chan & 1u][(ras_swap >> 6u) & 3u], ka);
        let aop = (w1 >> 16u) & 0xfu;
        var ares = tev_op1(aop, aa, ab, ac, ad, (w1 >> 20u) & 3u, (w1 >> 22u) & 3u);
        if (((w1 >> 24u) & 1u) != 0u) {
            ares = clamp(ares, 0.0, 1.0);
        } else {
            ares = clamp(ares, -4.0, 4.0);
        }
        let aoutreg = (w1 >> 25u) & 3u;
        regs[aoutreg].a = ares;
    }

    var prev = regs[0];

    if (ubuf.fog_type != 7u) {
        let z = 1.0 - in.pos.z; // reversed Z
        let fog_f = clamp((ubuf.fog.a / (ubuf.fog.b - z)) - ubuf.fog.c, 0.0, 1.0);
        var fog_z = 0.0;
        switch ubuf.fog_type {
            case 0u: { fog_z = fog_f; }
            case 1u: { fog_z = 1.0 - exp2(-8.0 * fog_f); }
            case 2u: { fog_z = 1.0 - exp2(-8.0 * fog_f * fog_f); }
            case 3u: { fog_z = exp2(-8.0 * (1.0 - fog_f)); }
            default: {
                let f = 1.0 - fog_f;
                fog_z = exp2(-8.0 * f * f);
            }
        }
        prev = vec4f(mix(prev.rgb, ubuf.fog.color.rgb, clamp(fog_z, 0.0, 1.0)), prev.a);
    }

    if (ubuf.alpha_comp != 0xffffffffu) {
        let a8 = u32(round(clamp(prev.a, 0.0, 1.0) * 255.0));
        let c0 = alpha_cmp(a8, ubuf.alpha_comp & 7u, (ubuf.alpha_comp >> 8u) & 0xffu);
        let c1 = alpha_cmp(a8, (ubuf.alpha_comp >> 3u) & 7u, (ubuf.alpha_comp >> 16u) & 0xffu);
        let op = (ubuf.alpha_comp >> 6u) & 3u;
        var pass_test = false;
        switch op {
            case 0u: { pass_test = c0 && c1; }
            case 1u: { pass_test = c0 || c1; }
            case 2u: { pass_test = c0 != c1; }
            default: { pass_test = c0 == c1; }
        }
        if (!pass_test) {
            discard;
        }
    }
    return prev;
}
)""";

// ---------------------------------------------------------------------------

namespace {
// Texcoords referenced by any TEV stage's sampling.
u32 referenced_texcoords(const ShaderConfig& config) {
  u32 mask = 0;
  for (u32 i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    if (stage.texMapId <= GX_TEXMAP7 && stage.texCoordId <= GX_TEXCOORD7) {
      mask |= 1u << static_cast<u32>(stage.texCoordId);
    }
  }
  return mask;
}
} // namespace

bool supports(const ShaderConfig& config) noexcept {
  if (config.lineMode != 0 || config.numIndStages != 0 || config.tevStageCount == 0) {
    return false;
  }
  for (u32 i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    switch (stage.channelId) {
    case GX_COLOR0A0:
    case GX_COLOR1A1:
    case GX_COLOR_ZERO:
    case GX_COLOR_NULL:
      break;
    default:
      return false; // alpha bump channels etc.
    }
    const bool samplesTex = stage.colorPass.a == GX_CC_TEXC || stage.colorPass.b == GX_CC_TEXC ||
                            stage.colorPass.c == GX_CC_TEXC || stage.colorPass.d == GX_CC_TEXC ||
                            stage.colorPass.a == GX_CC_TEXA || stage.colorPass.b == GX_CC_TEXA ||
                            stage.colorPass.c == GX_CC_TEXA || stage.colorPass.d == GX_CC_TEXA ||
                            stage.alphaPass.a == GX_CA_TEXA || stage.alphaPass.b == GX_CA_TEXA ||
                            stage.alphaPass.c == GX_CA_TEXA || stage.alphaPass.d == GX_CA_TEXA;
    if (samplesTex && (stage.texMapId > GX_TEXMAP7 || stage.texCoordId > GX_TEXCOORD7)) {
      return false;
    }
  }
  const u32 tcMask = referenced_texcoords(config);
  for (u32 i = 0; i < 8; ++i) {
    if ((tcMask & (1u << i)) == 0) {
      continue;
    }
    const auto& tcg = config.tcgs[i];
    if (tcg.type != GX_TG_MTX2x4 && tcg.type != GX_TG_MTX3x4) {
      return false;
    }
    if (tcg.postMtx != GX_PTIDENTITY) {
      return false;
    }
    const bool srcOk =
        tcg.src == GX_TG_POS || tcg.src == GX_TG_NRM || (tcg.src >= GX_TG_TEX0 && tcg.src <= GX_TG_TEX7);
    if (!srcOk) {
      return false;
    }
  }
  for (int a = GX_VA_PNMTXIDX; a <= GX_VA_TEX7; ++a) {
    const auto& attr = config.attrs[a];
    if (attr.attrType == GX_NONE) {
      continue;
    }
    if (attr.nbt3 || attr.cnt > 4) {
      return false;
    }
  }
  for (int c = 0; c < MaxColorChannels; ++c) {
    const auto& cc = config.colorChannels[c];
    if (cc.ambSrc != GX_SRC_REG && cc.ambSrc != GX_SRC_VTX) {
      return false;
    }
    if (cc.matSrc != GX_SRC_REG && cc.matSrc != GX_SRC_VTX) {
      return false;
    }
  }
  return true;
}

namespace {
u32 encode_comp_type(GXAttr attr, u8 compType) {
  if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
    return 5u + compType; // GX_RGB565=0 .. GX_RGBA8=5 -> codes 5..10
  }
  return compType; // GX_U8=0 .. GX_F32=4
}

u32 encode_fog(u8 fogType) {
  switch (fogType) {
  case GX_FOG_PERSP_LIN:
  case GX_FOG_ORTHO_LIN:
    return 0;
  case GX_FOG_PERSP_EXP:
  case GX_FOG_ORTHO_EXP:
    return 1;
  case GX_FOG_PERSP_EXP2:
  case GX_FOG_ORTHO_EXP2:
    return 2;
  case GX_FOG_PERSP_REVEXP:
  case GX_FOG_ORTHO_REVEXP:
    return 3;
  case GX_FOG_PERSP_REVEXP2:
  case GX_FOG_ORTHO_REVEXP2:
    return 4;
  default:
    return 7;
  }
}
} // namespace

gfx::Range build_uniform(const ShaderConfig& config, u32 vtxStart, const BindGroupRanges& ranges) noexcept {
  flush_matrix_palette();

  static ByteBuffer buf;
  buf.clear();

  // header
  buf.append(vtxStart);
  buf.append(g_gxState.currentPnMtx);
  buf.append<u32>(config.vtxStride);
  buf.append<u32>(config.tevStageCount);
  buf.append<u32>(8); // num_texgens (unused; tcg words carry a valid bit)
  buf.append<u32>(encode_fog(config.fogType));
  if (config.alphaCompare) {
    buf.append<u32>(static_cast<u32>(config.alphaCompare.comp0) | (static_cast<u32>(config.alphaCompare.comp1) << 3) |
                    (static_cast<u32>(config.alphaCompare.op) << 6) |
                    (static_cast<u32>(config.alphaCompare.ref0) << 8) |
                    (static_cast<u32>(config.alphaCompare.ref1) << 16));
  } else {
    buf.append<u32>(UINT32_MAX);
  }
  u32 swapTables = 0;
  for (int t = 0; t < MaxTevSwap; ++t) {
    const auto& sw = config.tevSwapTable[t];
    swapTables |= (static_cast<u32>(sw.red) | (static_cast<u32>(sw.green) << 2) | (static_cast<u32>(sw.blue) << 4) |
                   (static_cast<u32>(sw.alpha) << 6))
                  << (t * 8);
  }
  buf.append(swapTables);
  // tex_uv_map: texcoord used per texture map (first stage referencing it)
  u32 texUvMap = 0;
  for (u32 i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    if (stage.texMapId <= GX_TEXMAP7 && stage.texCoordId <= GX_TEXCOORD7) {
      const u32 map = static_cast<u32>(stage.texMapId);
      if (((texUvMap >> (map * 4)) & 0x8u) == 0) {
        texUvMap |= (static_cast<u32>(stage.texCoordId) | 0x8u) << (map * 4);
      }
    }
  }
  // strip the "assigned" marker bits back out
  u32 texUvMapClean = 0;
  for (u32 map = 0; map < 8; ++map) {
    texUvMapClean |= ((texUvMap >> (map * 4)) & 0x7u) << (map * 4);
  }
  buf.append(texUvMapClean);
  buf.append<u32>(0);
  buf.append<u32>(0);
  buf.append<u32>(0);

  // array_start (12 used, pad to vec4u x3)
  for (const auto& vaRange : ranges.vaRanges) {
    buf.append<u32>(vaRange.offset);
  }

  // mtx_start (32)
  buf.append(g_gxState.mtxOffsets);

  // attrs (21 used, pad to 24)
  for (int a = GX_VA_PNMTXIDX; a <= GX_VA_TEX7; ++a) {
    const auto& attr = config.attrs[a];
    u32 w = 0;
    if (attr.attrType != GX_NONE) {
      w = (attr.attrType & 3u) | ((attr.cnt & 7u) << 2) |
          ((encode_comp_type(static_cast<GXAttr>(a), attr.compType) & 0xfu) << 5) | ((attr.frac & 0x1fu) << 9) |
          ((attr.le ? 1u : 0u) << 14) | ((static_cast<u32>(attr.offset) & 0xffu) << 16) |
          ((static_cast<u32>(attr.stride) & 0xffu) << 24);
    }
    buf.append(w);
  }
  buf.append<u32>(0);
  buf.append<u32>(0);
  buf.append<u32>(0);

  // tcgs (8)
  const u32 tcMask = referenced_texcoords(config);
  for (u32 i = 0; i < 8; ++i) {
    const auto& tcg = config.tcgs[i];
    u32 w = 0;
    if ((tcMask & (1u << i)) != 0 && (tcg.type == GX_TG_MTX2x4 || tcg.type == GX_TG_MTX3x4)) {
      const u32 type = tcg.type == GX_TG_MTX3x4 ? 1u : 0u;
      u32 src;
      if (tcg.src == GX_TG_POS) {
        src = 0;
      } else if (tcg.src == GX_TG_NRM) {
        src = 1;
      } else {
        src = 2 + (tcg.src - GX_TG_TEX0);
      }
      const u32 slot = tcg.mtx == GX_IDENTITY ? 31u : static_cast<u32>(tcg.mtx) / 3u;
      const bool hasIdx = config.attrs[GX_VA_TEX0MTXIDX + i].attrType == GX_DIRECT;
      w = type | (src << 2) | ((slot & 0x1fu) << 6) | ((hasIdx ? 1u : 0u) << 11) | ((tcg.normalize ? 1u : 0u) << 12) |
          (1u << 13);
    }
    buf.append(w);
  }

  // channel words (color0, color1, alpha0, alpha1)
  for (int c = 0; c < 4; ++c) {
    const int idx = c < 2 ? c : (c - 2) + GX_ALPHA0;
    const auto& cc = config.colorChannels[idx];
    u32 diffFn = cc.diffFn == GX_DF_SIGN ? 1u : (cc.diffFn == GX_DF_CLAMP ? 2u : 0u);
    u32 attnFn = cc.attnFn == GX_AF_SPOT ? 1u : (cc.attnFn == GX_AF_SPEC ? 2u : 0u);
    buf.append<u32>((cc.lightingEnabled ? 1u : 0u) | ((cc.ambSrc == GX_SRC_VTX ? 1u : 0u) << 1) |
                    ((cc.matSrc == GX_SRC_VTX ? 1u : 0u) << 2) | (diffFn << 3) | (attnFn << 5));
  }

  // stages (16)
  for (u32 i = 0; i < MaxTevStages; ++i) {
    if (i >= config.tevStageCount) {
      buf.append<u32>(0);
      buf.append<u32>(0);
      buf.append<u32>(0);
      buf.append<u32>(0);
      continue;
    }
    const auto& st = config.tevStages[i];
    const u32 w0 = (static_cast<u32>(st.colorPass.a) & 0xfu) | ((static_cast<u32>(st.colorPass.b) & 0xfu) << 4) |
                   ((static_cast<u32>(st.colorPass.c) & 0xfu) << 8) | ((static_cast<u32>(st.colorPass.d) & 0xfu) << 12) |
                   ((static_cast<u32>(st.alphaPass.a) & 7u) << 16) | ((static_cast<u32>(st.alphaPass.b) & 7u) << 19) |
                   ((static_cast<u32>(st.alphaPass.c) & 7u) << 22) | ((static_cast<u32>(st.alphaPass.d) & 7u) << 25);
    const u32 w1 = (static_cast<u32>(st.colorOp.op) & 0xfu) | ((static_cast<u32>(st.colorOp.bias) & 3u) << 4) |
                   ((static_cast<u32>(st.colorOp.scale) & 3u) << 6) | ((st.colorOp.clamp ? 1u : 0u) << 8) |
                   ((static_cast<u32>(st.colorOp.outReg) & 3u) << 9) | ((static_cast<u32>(st.alphaOp.op) & 0xfu) << 16) |
                   ((static_cast<u32>(st.alphaOp.bias) & 3u) << 20) | ((static_cast<u32>(st.alphaOp.scale) & 3u) << 22) |
                   ((st.alphaOp.clamp ? 1u : 0u) << 24) | ((static_cast<u32>(st.alphaOp.outReg) & 3u) << 25);
    u32 chan;
    switch (st.channelId) {
    case GX_COLOR0A0:
      chan = 0;
      break;
    case GX_COLOR1A1:
      chan = 1;
      break;
    default:
      chan = 2;
      break;
    }
    const u32 texMap = st.texMapId <= GX_TEXMAP7 ? static_cast<u32>(st.texMapId) : 15u;
    const u32 texCoord = st.texCoordId <= GX_TEXCOORD7 ? static_cast<u32>(st.texCoordId) : 15u;
    const u32 w2 = (static_cast<u32>(st.kcSel) & 0x1fu) | ((static_cast<u32>(st.kaSel) & 0x1fu) << 5) | (chan << 10) |
                   (texCoord << 12) | (texMap << 16) | ((static_cast<u32>(st.tevSwapTex) & 3u) << 20) |
                   ((static_cast<u32>(st.tevSwapRas) & 3u) << 22);
    buf.append(w0);
    buf.append(w1);
    buf.append(w2);
    buf.append<u32>(0);
  }

  // state
  buf.append(g_gxState.proj);
  for (int i = 0; i < MaxTevRegs; ++i) {
    buf.append(g_gxState.colorRegs[i]);
  }
  for (int i = 0; i < GX_MAX_KCOLOR; ++i) {
    buf.append(g_gxState.kcolors[i]);
  }
  for (int c = 0; c < 4; ++c) {
    const int idx = c < 2 ? c : (c - 2) + GX_ALPHA0;
    buf.append(g_gxState.colorChannelState[idx].ambColor);
  }
  for (int c = 0; c < 4; ++c) {
    const int idx = c < 2 ? c : (c - 2) + GX_ALPHA0;
    buf.append(g_gxState.colorChannelState[idx].matColor);
  }
  for (int c = 0; c < 4; ++c) {
    const int idx = c < 2 ? c : (c - 2) + GX_ALPHA0;
    buf.append<u32>(static_cast<u32>(g_gxState.colorChannelState[idx].lightMask.to_ulong()));
  }
  buf.append(g_gxState.lights);
  {
    const auto& fog = g_gxState.fog;
    buf.append(fog.color);
    buf.append(fog.a);
    buf.append(fog.b);
    buf.append(fog.c);
    buf.append(0.f);
  }
  for (int i = 0; i < 8; ++i) {
    const auto& tex = get_texture(static_cast<GXTexMapID>(i));
    buf.append(texture_size_bias(tex));
  }

  return gfx::push_uniform(buf.data(), buf.size());
}

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config) {
  static wgpu::ShaderModule module;
  static std::once_flag once;
  std::call_once(once, [] {
    const wgpu::ShaderSourceWGSL wgslSource{wgpu::ShaderSourceWGSL::Init{
        .code = UberShaderSource,
    }};
    const wgpu::ShaderModuleDescriptor descriptor{
        .nextInChain = &wgslSource,
        .label = "GX Uber Shader",
    };
    module = g_device.CreateShaderModule(&descriptor);
    Log.info("compiled uber shader module");
  });
  return build_pipeline(config, {}, module, "GX Uber Pipeline");
}

gfx::PipelineRef pipeline_ref(const PipelineConfig& config) {
  return gfx::find_pipeline(gfx::ShaderType::GXUber, config, [=] { return uber::create_pipeline(config); });
}

} // namespace aurora::gx::uber
