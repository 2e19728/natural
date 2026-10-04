#pragma once

#include <cstdint>
#include <cstring>
#include <utility>
#include "basic_arithmetics.h"
#include "array_u64.h"

// C++ inline replacements for asmDiv / asmMod / asmMulMod (see mod_arith.h):
//   asmDiv(Low, High, Mod)      -> div_128_64(Low, High, Mod)
//   asmMod(A, R, Mod)           -> barrett_mod(A, R, Mod)          R = ceil(2^64  / Mod)
//   asmMulMod(A, B, Mod, R)     -> barrett_mul_mod(A, B, Mod, R)   R = ceil(2^124 / Mod)
#include "barrett_reduction.h"

namespace nat {

extern "C" {

	void nat_asmNtt(uint64_t _N, uint64_t _Mod, uint64_t* _Begin, const uint64_t* _Root, uint64_t* _End);

	void nat_asmNtt2(uint64_t _N, uint64_t _Mod, uint64_t* _Begin, const uint64_t* _Root, uint64_t* _End);

	void nat_asmINtt(uint64_t _N, uint64_t _Mod, uint64_t* _Begin, const uint64_t* _Root, uint64_t* _End);

	void nat_asmINtt2(uint64_t _N, uint64_t _Mod, uint64_t* _Begin, const uint64_t* _Root, uint64_t* _End);





	// Fused last NTT layer x pointwise multiply x first INTT layer, for modulus _I
	// (one generic function replacing the original asmNttMul0/1/2; see mul_ntt.s).
	void nat_asmNttMul(uint64_t _N, uint64_t* _Dst, const uint64_t* _Src1, const uint64_t* _Src2, const uint64_t* _Root, uint64_t _I);

	// CRT-merge _Size coefficients of the three residue arrays into 64-bit words
	// (data_0 in place), leaving the two carry words in data_1[0], data_1[1] and
	// returning the middle one (see mul_ntt.s).
	uint64_t nat_asmCRT(uint64_t* _Data_0, uint64_t* _Data_1, uint64_t* _Data_2, uint64_t _Size);
}

// The two kernels that used to be assembly (asmLoad and asmINttShr, both formerly in
// ntt/optimized/ntt_info.s) are now C++: the asmLoad folding (Dst1[i] = Dst2[i] =
// Src[i] mod Mod, one barrett_mod step per limb) is open-coded in ntt_workspace::load(), and the
// fused last inverse layer + right shift is intt_shr() below.  Both were checked to be
// bit-for-bit equivalent to the assembly they replace (bench/bench_load.cpp,
// bench/bench_shrink.cpp, bench/diag_ntt.cpp, and the unchanged v0/v1/v2 hashes).
//
// INTT()'s last layer + right shift: butterflies at distance h = 2^(_Scale-1) fused
// with the division by 2^(_Scale-1) ("Shr" = the final right shift; the inverse layers
// here are unnormalized radix-2 and there are _Scale-1 of them, so the total scale
// factor is 2^(_Scale-1) -- NOT 2^_Scale).  This replaces asmINttShr.
//
// Layout handled by one iteration (j = 0, 2, 4, ... , h-2):
//
//	a0 = b[j]      a1 = b[j+1]           (first half)
//	c0 = b[j+h]    c1 = b[j+h+1]         (second half)
//	b[j]      = shr(reduce(a0 + c0))     b[j+h]   = shr(reduce(a0 - c0))
//	b[j+1]    = shr(reduce(a1 + c1))     b[j+h+1] = shr(reduce(a1 - c1))
//
// add/sub are single conditional corrections (inputs < Mod < 2^63, so a + c < 2^64 and
// a - c > -(2^63) cannot overflow).  shr() does the fused shift by s = _Scale-1; it is
// valid because Mod = (Mod >> s) << s | 1 for every s <= 56 (Mod = alpha*2^56 + 1):
//
//	shr(v) = ((int64_t)(v-1) >> s) + 1 + ((h-1) & ~(v-1)) * (Mod >> s)
//
// Proof: for v >= 1 put u = v-1 = q*2^s + r, 0 <= r < 2^s.  Then (h-1) & ~u = 2^s-1-r
// and, modulo Mod (using 2^s*(Mod >> s) = Mod - 1),
//
//	2^s * shr(v) = 2^s*q + 2^s + (2^s-1-r)*(Mod-1)
//	             == u - r + 2^s - (2^s-1-r) = u + 1 = v   (mod Mod).
//
// For v = 0 the arithmetic shift gives ((int64_t)-1 >> s) = -1, cancelled by the +1,
// and the mask term vanishes because ~(v-1) = 0: shr(0) = 0.  Range: 0 <= shr(v) < Mod
// (maximum 2^s*(Mod >> s) = Mod-1, reached at v = 1), so the output is a fully reduced
// residue, exactly as the assembly leaves it.
inline uint64_t intt_shr_step(uint64_t _V, unsigned _S, uint64_t _Mask, uint64_t _Mr) {
	uint64_t _U = _V - 1;
	return (uint64_t)((int64_t)_U >> _S) + 1 + ((_Mask & ~_U) * _Mr);
}

inline void intt_shr(int _Scale, uint64_t _Mod, uint64_t* _Begin) {
	const unsigned s = (unsigned)_Scale - 1;
	const uint64_t h = 1ull << s;             // butterfly distance of the fused layer
	const uint64_t mask = h - 1;
	const uint64_t mr = _Mod >> s;            // Mod = mr * 2^s + 1
	uint64_t* lo = _Begin;
	uint64_t* hi = _Begin + h;
	for (uint64_t i = h >> 1; i; i--, lo += 2, hi += 2) {
		uint64_t a0 = lo[0], a1 = lo[1];
		uint64_t c0 = hi[0], c1 = hi[1];

		uint64_t s0 = a0 + c0, s1 = a1 + c1;
		uint64_t d0 = a0 - c0, d1 = a1 - c1;
		if (s0 >= _Mod) s0 -= _Mod;
		if (s1 >= _Mod) s1 -= _Mod;
		if ((int64_t)d0 < 0) d0 += _Mod;
		if ((int64_t)d1 < 0) d1 += _Mod;

		lo[0] = intt_shr_step(s0, s, mask, mr);
		lo[1] = intt_shr_step(s1, s, mask, mr);
		hi[0] = intt_shr_step(d0, s, mask, mr);
		hi[1] = intt_shr_step(d1, s, mask, mr);
	}
}

struct ntt_workspace {

	static const uint64_t mods[3];
	static const uint64_t mods_124[3];
	static const uint64_t root_base[3][56];
	static const uint64_t iroot_base[3][56];
	static array_u64 root[3];
	static array_u64 iroot[3];
	static int root_scale;
	array_u64 ntt_data[3];
	int ntt_scale;

	static void init(int scale) {
		scale -= 1;
		if (scale <= root_scale)
			return;
		for (int i = 0; i < 3; i++) {
			root[i].resize(1ull << scale);
			for (int j = root_scale; j < scale; j++)
				for (uint64_t k = 0, n = 1ull << j; k < n; k += 2) {
					root[i][n + k] = barrett_mul_mod(root[i][k], root_base[i][j], mods[i], mods_124[i]);
					root[i][n + k + 1] = div_128_64(0, root[i][n + k], mods[i]) + 1;
				}
			iroot[i].resize(1ull << scale);
			for (int j = root_scale; j < scale; j++)
				for (uint64_t k = 0, n = 1ull << j; k < n; k += 2) {
					iroot[i][n + k] = barrett_mul_mod(iroot[i][k], iroot_base[i][j], mods[i], mods_124[i]);
					iroot[i][n + k + 1] = div_128_64(0, iroot[i][n + k], mods[i]) + 1;
				}
		}
		root_scale = scale;
	}

	ntt_workspace(int scale = 0) :ntt_scale(scale) {}

	ntt_workspace(const array_u64& n, int scale) {
		load(n, scale);
	}

	ntt_workspace(const uint64_t* src, uint64_t len, int scale) {
		load(src, len, scale);
	}

	void load(const uint64_t*, uint64_t, int);

	void load(const array_u64&, int);

	void ntt();

	void intt();

	void mul(const ntt_workspace&, const ntt_workspace&);

	uint64_t crt(uint64_t);

	void save(array_u64&, uint64_t);

};

inline const uint64_t ntt_workspace::mods[3] = {
	27ull << 56 | 1,
	58ull << 56 | 1,
	87ull << 56 | 1
};

inline const uint64_t ntt_workspace::mods_124[3] = {
	0x97b425ed097b425a,
	0x469ee58469ee5846,
	0x2f149902f149902f
};

inline const uint64_t ntt_workspace::root_base[3][56]{
	{
		0x1b00000000000000, 0x0c55f5e087b66679, 0x05d00f46ebf68efc, 0x048864e8f79c7099, 0x121a7906088f35c9, 0x0a707c27703307a8, 0x1a72390d129c217e, 0x14794b5b0eff3c1d,
		0x150aba0a482a0f41, 0x06724e24293f5b86, 0x0e4944f5f7cf4097, 0x186725790196b6d7, 0x1282d5a8cd67201f, 0x12e9e778e733b261, 0x02b70f6b0d4a2680, 0x059df5b4354e9d7f,
		0x193aa62399495777, 0x11b8a30797545f26, 0x00014f5c30f8efbc, 0x05ed6f8b2ed1d536, 0x0906d3b9852ec38f, 0x10c45314cc77b053, 0x18275e18005bcf48, 0x0514d6456c5a8365,
		0x16a4e6b4538c15c6, 0x19dc8ebd3864bb9c, 0x1253c4291e14513f, 0x03390b0b8a8f39ef, 0x115460f2af355b5d, 0x03e7d4c6519e4fae, 0x1612bd0b0c807014, 0x19b48989659bbc1e,
		0x18c8ad8701dce021, 0x00656b06b1fef5e1, 0x192054d8e0d98368, 0x00f76bace7abd4cb, 0x07768878a81ffa6c, 0x028c219d139bbe02, 0x0959bbe081d127fb, 0x13bd1ce230a76d3f,
		0x144f394db1ebb988, 0x09a7965810a40b98, 0x07c62c5c31a01917, 0x198264d7b92a0ff5, 0x07f27550e89dfee7, 0x0aa7be203ac5edf6, 0x0b3299e03fa84a13, 0x0250dc45636976a8,
		0x18e5a8d2f0982456, 0x17c26444df30c6a7, 0x17fd02a7837348ff, 0x00a33f092e0b1ac1, 0x000000000cc6db61, 0x0000000000003931, 0x0000000000000079, 0x000000000000000b
	},
	{
		0x3a00000000000000, 0x0b5ea5c01435f877, 0x279e926c244e5b38, 0x120da1843a627638, 0x24a8f0822831956e, 0x2199348392e64bed, 0x1778454b4655688a, 0x29e345ba8d50ff33,
		0x28037c699a8409bf, 0x02946bccfb7875a5, 0x19a702af56d79371, 0x0c88fcc933e3ade0, 0x2d04aabf083e596d, 0x2c7c83ddc4e6045d, 0x1ed47d8d9fe4383e, 0x221350494f6cbd85,
		0x0f979b66cc0978e8, 0x19ad9482e86163cc, 0x01ed78e2f5f9a9dc, 0x237efa2c5e3e99ae, 0x2cf497d4f2918a3c, 0x0ebb70fd1d2ce835, 0x30019ed71781c420, 0x0977b9196e263958,
		0x2b20e0cfec908df3, 0x2356b574c520834d, 0x139fa4cbb31bcb02, 0x051b9372b32d4045, 0x0b8fd90431a61c6a, 0x32d6f90bf41b8a2a, 0x008393580a8f8d40, 0x12ab3e189876c2b6,
		0x1eff4afb813fb1e1, 0x1b76615f315173a1, 0x21ce3fe2322a21cc, 0x3995503c81e90438, 0x173348425a387d2c, 0x22781b63e914d58b, 0x1653f406f9de7aa8, 0x0398d957310a8395,
		0x098ac5d4b01acf8f, 0x0c4bccdfe12d0ff6, 0x32c488ca19c62e55, 0x201dc82ac3c4a9ad, 0x11f45747e6fd15d5, 0x14eeab6b3026e680, 0x11b95283b1adf675, 0x2b50093ca8d38830,
		0x1a11a7b9611a7baa, 0x17fffffffffffffc, 0x0000000100000000, 0x0000000000010000, 0x0000000000000100, 0x0000000000000010, 0x0000000000000004, 0x0000000000000002
	},
	{
		0x5700000000000000, 0x25eeaecc59203257, 0x0894a6a0d69ffa9f, 0x45efe465dd261a8c, 0x32004bea8d8c6b05, 0x51d1a4a056db8efb, 0x030e0e60a108c7d4, 0x1552e0364c6698c6,
		0x1fde67485112f101, 0x121d2babe5a17006, 0x40ae7411f8b2079c, 0x049bd54e09b7bcc0, 0x5618c746b86b3fff, 0x014f9a3b3c536562, 0x478b136dc0acb53c, 0x0cd928ca349150a7,
		0x2aac61c39caada38, 0x0a2f8dfe14850f1a, 0x251a8be905833450, 0x3c7dfa086e287454, 0x0a2700365877fc92, 0x1f874ec09c259bc9, 0x1d7a0653ae3ab538, 0x561da2e6aca4bb66,
		0x3745007b4c5f4e96, 0x52d8076b73305796, 0x5609b51e6262b905, 0x450fb776ccea8260, 0x1209fb38bbce4cbd, 0x3b91a96863b41e6d, 0x2a805f5540305fd1, 0x18d074f21d9bb9c7,
		0x27852f125ce038c5, 0x27a13c78a2cea5b1, 0x39bd7b08d38f1b5a, 0x28f42c1355f5a4d5, 0x562c74f137904ab4, 0x15c0b5221862604a, 0x3e45003fe6672245, 0x21a3aa11c512f4f4,
		0x1194f148ad0d95d0, 0x1b10f7a7211a7631, 0x12e9d86a51add86a, 0x0ebc8981589d2c1c, 0x29ebf23aba3a515a, 0x4088ee8ac148c260, 0x0665ef7befb47442, 0x03b70fe5e9d5ef2f,
		0x3d9941c499089e44, 0x00dea4a7c31535f8, 0x28a24eadb9b14771, 0x0b43389cfed1ac42, 0x349fc3f96684e0f3, 0x0000000226761f10, 0x0000000000017764, 0x0000000000000136
	}
};

inline const uint64_t ntt_workspace::iroot_base[3][56]{
	{
		0x1b00000000000000, 0x0eaa0a1f78499988, 0x181f53f62f0a3966, 0x11af8e382966df93, 0x061aa49bc8c68e7e, 0x0ee4b3f720abbc84, 0x188ae203558327dc, 0x01601d934ca27e0e,
		0x1a44188b632c0036, 0x1094e3ae85c8a861, 0x0a9690e4cfe7b4e1, 0x03f64170fcbd6640, 0x080062ecea3a9cae, 0x19194e8d40b9665e, 0x18978ad2c429fac8, 0x04359ca87bd86a30,
		0x0f9b726e21b3f742, 0x0f1a783886569e6d, 0x1210471efff7de08, 0x1a3522951787cc05, 0x1aaf36ec832c9fa0, 0x06352217d4cbdfc4, 0x0590d57bde4a8e03, 0x11b4d8443bc4f25f,
		0x118c9ac8cd5018b1, 0x0739f4d44ff6d1ed, 0x0bdff3199cff1ca3, 0x0f36409b45b6d00c, 0x007bc2bce66c1eff, 0x0807524f30ede847, 0x129bd036bc2e03c7, 0x0166d6b5e67524ab,
		0x0435db457f1262f1, 0x06cd41522df9f8df, 0x02d206ac543db691, 0x15c46158b49a22ff, 0x018a56a0143c8298, 0x03b8b55bd55f90ad, 0x0d612fc0ab420270, 0x107e4118998c1296,
		0x1103280e9f950ba2, 0x1295913d096a25a0, 0x0395629478b99b29, 0x1944918378622eef, 0x0f9982e637edfdb7, 0x02ce6b02d6a1872d, 0x18af972937bf8837, 0x0820108a39a973a0,
		0x10bde1b63a752e8c, 0x1033cd166401aab4, 0x1a6dfc09a947a753, 0x1444a755ef8bbf37, 0x0edbbe88bf7a288d, 0x027beaced9690884, 0x0391fbc4c2a50659, 0x0c45d1745d1745d2
	},
	{
		0x3a00000000000000, 0x2ea15a3febca078a, 0x0bdc45c04b8fea45, 0x276b8974dd6cdc4b, 0x13491ee45cc16371, 0x0a55f976dd0f93fa, 0x020e75f34ca83188, 0x2948df4f456f48ad,
		0x334163ca861755a4, 0x2414c7c748f3358a, 0x3680ff448a1a1d94, 0x0721b0eb7782c0f9, 0x32ab7d5a2d8c4bdb, 0x124b9d8dabda2670, 0x0ba2f77129c3f56e, 0x21e162fe7455d1d0,
		0x071e52b626ad5fe2, 0x3044afc1b0ff9f47, 0x194bec9b5c636af3, 0x0236f8c90a01c927, 0x1ce82a257e5c870f, 0x252a46b37cafb952, 0x029433e8898337bc, 0x212af2916b6d4731,
		0x0051236b350ba95f, 0x1581987447b66c83, 0x3991bb1a88d0bacd, 0x262618747086d2ea, 0x337c79fabeaabb7b, 0x08bb669a723c6e01, 0x0b7877d255733d49, 0x1bd9a18450e0ae83,
		0x139c359d42adfa0a, 0x0179518bbea5d58a, 0x147d37867c24583e, 0x060e1415f841adbb, 0x12efeb6ae120a57d, 0x06d0227483c4901c, 0x0aa13a79cf0f1c3b, 0x0e972fc06a84d391,
		0x0467aca2354d1bed, 0x2205a6ed9b1f231e, 0x34352dfbb1d2a12b, 0x09e127647f9576b0, 0x0ffe32bb93cbe3fe, 0x2c1d6a47fee6749c, 0x39ffe59c956f4f87, 0x39d8e0ca60000001,
		0x3705d80000000001, 0x0d24000000000000, 0x39ffffffc6000001, 0x39ffc60000000001, 0x39c6000000000001, 0x3660000000000001, 0x2b80000000000001, 0x1d00000000000001
	},
	{
		0x5700000000000000, 0x31115133a6dfcdaa, 0x264f7a44f2dc101c, 0x024934e3f187ae56, 0x110156e0bc017643, 0x124be95ccbce10eb, 0x04c014168266d94f, 0x25264b4c5bcefe32,
		0x29b20b4bf5e85061, 0x4fd88af336c94b67, 0x0f1e455645baef59, 0x28eebb29fa488818, 0x41a96e76e1fc459e, 0x4bbd62d0824d6881, 0x348ba85808b6eccd, 0x10b1be7add3a6105,
		0x4be34f24f3f83aaf, 0x32d6740121c1900f, 0x080d66b4fd153f16, 0x1760865185aaae99, 0x2cb22191d6e6d69f, 0x446433be011ffcdf, 0x34b6dfd3d404e4d3, 0x12706af4051187d3,
		0x371890c6baf59540, 0x1a1b1b90377e6f48, 0x541c2553b9aaeb8d, 0x0ea761b6d90072ec, 0x0b4412168e0d4285, 0x0dd0736bdb64b6ad, 0x5410c9b99b145158, 0x0a5cb629a7d0d0d0,
		0x405308b924442778, 0x0e545025d3f8f6ec, 0x22144b9c76de03e4, 0x00f34806cd3f5f27, 0x1255f46f8bf9890d, 0x328a388d106657d3, 0x0e23b6553922bb6c, 0x05d954963798ff25,
		0x02725ea9834db7dc, 0x27bff1eb8e072dd0, 0x16046df4722c4c07, 0x44f69e898dbd4049, 0x0c706c58aeed540f, 0x4ee91287e5b7aae9, 0x3d6e5b538638db46, 0x0d4dde910b2cc1b9,
		0x1fdc9191dfa0fa07, 0x320f3c38a06ef140, 0x491d76bf22140b04, 0x5070424c82c3ab2e, 0x40bf123e57353d3e, 0x413f4490baa97eb6, 0x4926c9f732d0ff91, 0x38f8915789157892
	}
};

inline array_u64 ntt_workspace::root[3] = { { 1, 10 }, { 1, 5 }, { 1, 3 } };

inline array_u64 ntt_workspace::iroot[3] = { { 1, 10 }, { 1, 5 }, { 1, 3 } };

inline int ntt_workspace::root_scale = 1;

inline void ntt_workspace::load(const uint64_t* src, uint64_t len, int scale) {
	static constexpr uint64_t mods_64[3] = { 10, 5, 3 };
	ntt_scale = scale;
	uint64_t half_ntt_size = 1ull << (scale - 1);
	for (int i = 0; i < 3; i++)
		ntt_data[i].resize(half_ntt_size << 1);
	if (len <= half_ntt_size) {
		for (int i = 0; i < 3; i++) {
			uint64_t j = 0;
			for (; j < len; j++) {
				uint64_t n0 = barrett_mod(src[j], mods_64[i], mods[i]);
				ntt_data[i][j] = n0;
				ntt_data[i][j + half_ntt_size] = n0;
			}
			memset(ntt_data[i].data + len, 0, sizeof(uint64_t) * (half_ntt_size - len));
			memset(ntt_data[i].data + half_ntt_size + len, 0, sizeof(uint64_t) * (half_ntt_size - len));
		}
	}
	else {
		for (int i = 0; i < 3; i++) {
			uint64_t j = 0;
			for (; j < len - half_ntt_size; j++) {
				uint64_t n0 = barrett_mod(src[j], mods_64[i], mods[i]);
				uint64_t n1 = barrett_mod(src[j + half_ntt_size], mods_64[i], mods[i]);
				ntt_data[i][j] = n0 + n1;
				ntt_data[i][j + half_ntt_size] = n0 - n1 + mods[i];
			}
			for (; j < half_ntt_size; j++) {
				uint64_t n0 = barrett_mod(src[j], mods_64[i], mods[i]);
				ntt_data[i][j] = n0;
				ntt_data[i][j + half_ntt_size] = n0;
			}
		}
	}
	init(ntt_scale);
}

inline void ntt_workspace::load(const array_u64& n, int scale) {
	load(n.data, n.size, scale);
}

// ================= NTT schedule: one plan, DRAM / L3 / L2 / L1 levels =============
//
// The transform is a fixed sequence of DIF layers at distances 2^(k-1), 2^(k-2), ...,
// 2^1 (k = ntt_scale, array size 2^k = 2*2^(k-1)):
//
//   layer 2^(k-1)  ("fold")                     fused into load()      [edge pass]
//   layers 2^(k-2) .. 2^1                       ntt()
//   layer 2^0 + pointwise + 1st inverse layer    fused into nat_asmNttMul
//   layers 2^1 .. 2^(k-2)                       intt()
//   layer 2^(k-1) + the 2^-(k-1) shift           fused into intt_shr()  [edge pass]
//
// Layer j (distance 2^j) consists of independent butterflies: block b covers
// [b*2^(j+1), (b+1)*2^(j+1)) and pairs element i with i+2^j using twiddle omega^b,
// read from the flat table as the pair (root[i][2b], root[i][2b+1]) = (omega^b,
// Shoup/Barrett reciprocal of omega^b).  The whole schedule rests on two facts:
//
//  (1) block-diagonal commutation: every layer with 2^(j+1) <= 2^T is block diagonal
//      with respect to a split of the array into 2^T-element chunks, so such layers
//      may be run chunk by chunk in any interleaving, provided every layer with a
//      larger distance has already been run.  Chunking never changes a twiddle.
//  (2) the twiddle cursor is purely positional: a pass over [base, base+2^T) must
//      start at table element 2*(base/2^(j+1)) = base>>j.  "Twiddle indexing" here
//      means exactly this offset arithmetic -- plus the rule that the kernels that
//      skip the block-0 multiply (nat_asmNtt) is used iff base == 0.
//
// Every layer is its own pass: a pass at layer j over [base, base+2^T) reads its
// twiddles from table element base>>j and writes the 2^j-pairs in place.
//
// The four levels of the memory hierarchy, coarse to fine: DRAM, L3, L2, L1.  Level T runs
// the layers that fit a chunk of 2^T limbs, i.e. every layer j <= T-1: layer j is block
// diagonal with respect to 2^(j+1)-element blocks (fact (1)), so its blocks fit iff
// 2^(j+1) <= 2^T.  DRAM is the whole array (a chunk of 2^k); L3/L2/L1 are the 2^20 / 2^16 /
// 2^12-limb working sets -- 8 MiB / 512 KiB / 32 KiB, because the three moduli are
// transformed one after another (the modulus loop is outside the schedule), so only one
// array is hot at a time.  They sit under the reference CPU's L3 (24 MiB), L2 (1.25 MiB)
// and L1d (48 KiB):
//
//      DRAM: the whole array       pairs whose block fits in no smaller chunk
//      L3:   2^l3-element chunks   (ntt_scale_l3_threshold)
//      L2:   2^l2-element chunks   (ntt_scale_l2_threshold)
//      L1:   2^l1-element chunks   (ntt_scale_l1_threshold)
//
// Fact (1) forces DRAM before L3 before L2 before L1 forward and the exact mirror inverse,
// and inside a level the layers are run chunk by chunk (chunk-major), so a chunk stays
// cache resident across the whole level instead of being re-streamed once per layer.  The
// cut points are clamped to the array size and then de-duplicated, so a small transform has
// fewer levels rather than empty ones: at scale k <= l1 all four collapse into DRAM and the
// transform is a single chunk-major walk over the whole array.
//
// A level is a plain range of layers, one pass per layer, run descending forward and
// ascending inverse; ntt_fwd_sched/ntt_inv_sched keep the two mirror images of each other.
//
// What is deliberately *not* here: the classic four-step / six-step FFT, i.e.
// interpreting the array as an N1 x N2 matrix, running N1 transforms of length N2,
// multiplying by w_N^(n1*k2), running N2 transforms of length N1 and folding the two
// transposes into load()/save().  It cannot be applied to this DIF: its algebra needs
// the chunk-local cascade to be a *standalone* length-2^T DFT plus a rank-1 twiddle
// matrix, but here a sub-problem keeps the global generator omega, so local block k of
// local layer j carries omega^(k + O/2^j) with O the chunk offset.  The correction
// omega^(O/2^j) depends on j and is therefore not one elementwise multiply.
// Counterexample (k=3, T=2, chunk at O=4): de-rotating layer j=1 needs
// d[i+2]/d[i] = omega (i=0,1), layer j=0 needs d[1]/d[0] = omega^2 and
// d[3]/d[2] = omega^3; together those give d[3]/d[1] = omega^2, while layer j=1
// demands omega -- inconsistent, so no diagonal renormalisation exists.  Folding the
// rotation into the twiddle table (what this schedule does) costs only the sequential
// table arcs (~2% of the data traffic at scale 23), whereas a four-step would spend a
// whole extra pass over the array on the twiddle multiply.  The useful half of the
// idea is already in place: the three edge passes (fold in load(), distance-1 layer in
// nat_asmNttMul, wrap fold in save()) are fused, so no pass over the array pays for them.

// Cut points of the schedule, as chunk exponents: the transform is run at four levels --
// DRAM (the whole array, no constant) and then the L3 / L2 / L1 working sets.  A level's
// chunk is 2^T limbs = 2^T * 8 bytes of resident data (one modulus at a time), which for the
// reference CPU gives T = 12 (32 KiB, L1d 48 KiB), 16 (512 KiB, L2 1.25 MiB) and 20 (8 MiB,
// L3 24 MiB).  Any set
// with 2 <= l1 <= l2 <= l3 is correct (the values are clamped and de-duplicated per scale),
// so a cut-point scan can override them without editing this file:
// -Dntt_scale_l2_threshold=14.
//
// inline constexpr, not plain `const`: a namespace-scope const object has INTERNAL
// linkage, i.e. one private copy per TU; `inline` gives it a single shared definition.
#ifndef ntt_scale_l1_threshold
inline constexpr int ntt_scale_l1_threshold = 12;
#endif
#ifndef ntt_scale_l2_threshold
inline constexpr int ntt_scale_l2_threshold = 16;
#endif
#ifndef ntt_scale_l3_threshold
inline constexpr int ntt_scale_l3_threshold = 20;
#endif

// One level of the schedule: the layer range [lo, hi] (layer j = distance 2^j, run
// descending forward and ascending inverse), one pass per layer.  hi < lo means the
// level is empty.
struct ntt_level {
	int lo, hi;    // lowest and highest layer of the level
};

// The schedule of one transform: at most four levels, DRAM, L3, L2 and L1.  chunk[0] is
// always k (the DRAM level runs over the whole array), so level s works on
// 2^chunk[s]-element chunks, of which there are 2^(chunk[s-1]-chunk[s]) per parent chunk.
struct ntt_sched {
	static constexpr int MAXLEVEL = 4;
	int nlevel;
	int chunk[MAXLEVEL];
	ntt_level lv[MAXLEVEL];
};

// Schedules layers 2^(k-2) .. 2^1 -- the 2^(k-1) fold is fused into load() and the 2^0
// layer into nat_asmNttMul.  Cut points are the cache thresholds, clamped to the array (a
// threshold above k adds nothing) and de-duplicated (several clamping to the same value
// collapse into one level).
inline ntt_sched ntt_sched_for(int k) {
	ntt_sched S;
	S.nlevel = 0;
	const int want[ntt_sched::MAXLEVEL] = {
		k, ntt_scale_l3_threshold, ntt_scale_l2_threshold, ntt_scale_l1_threshold
	};
	for (int i = 0; i < ntt_sched::MAXLEVEL; i++) {
		int v = want[i];
		if (v > k) v = k;
		if (v < 2) v = 2;
		if (S.nlevel > 0 && S.chunk[S.nlevel - 1] == v)
			continue;
		S.chunk[S.nlevel++] = v;
	}
	int lim = k - 1;
	int b[ntt_sched::MAXLEVEL];
	// boundary b[i]: level i runs layers [b[i+1], b[i]-1], with b[0] = k-1.  A boundary is
	// clamped to its chunk -- layer j is block diagonal with respect to 2^(j+1)-element
	// blocks, so it fits a 2^chunk chunk iff j <= chunk-1.  The clamp is what keeps
	// lv.hi <= chunk-1; the block counts in ntt_*_sched use S.chunk[], not b[].
	for (int i = 0; i < S.nlevel; i++) {
		if (lim > S.chunk[i]) lim = S.chunk[i];
		if (lim < 1) lim = 1;          // degenerate cut points can undershoot on tiny sizes
		b[i] = lim;
		lim--;
	}
	for (int i = 0; i < S.nlevel; i++) {
		ntt_level& lv = S.lv[i];
		lv.hi = b[i] - 1;                              // level i covers [b[i+1], b[i]-1]
		lv.lo = (i + 1 < S.nlevel) ? b[i + 1] : 1;
	}
	return S;
}

// One forward level over [base, base+2^span): one pass per layer, distances descending.
// The twiddle cursor stays positional -- a pass at layer j over this chunk starts at
// table element base>>j -- and the block-0 kernels are used iff base == 0.
inline void ntt_fwd_level(uint64_t* _Data, uint64_t _Base, uint64_t _Span, const ntt_level& lv, int _I) {
	if (lv.hi < lv.lo)
		return;
	const uint64_t* R = ntt_workspace::root[_I].data;
	const uint64_t M = ntt_workspace::mods[_I];
	uint64_t* B = _Data + _Base;
	uint64_t* E = B + _Span;
	for (int j = lv.hi; j >= lv.lo; j--) {
		const uint64_t d = 1ull << j;
		const uint64_t* r = R + (_Base >> j);
		if (_Base == 0) nat_asmNtt(d, M, B, r, E);
		else nat_asmNtt2(d, M, B, r, E);
	}
}

// Mirror of ntt_fwd_level: one pass per layer, distances ascending.
inline void ntt_inv_level(uint64_t* _Data, uint64_t _Base, uint64_t _Span, const ntt_level& lv, int _I) {
	if (lv.hi < lv.lo)
		return;
	const uint64_t* R = ntt_workspace::iroot[_I].data;
	const uint64_t M = ntt_workspace::mods[_I];
	uint64_t* B = _Data + _Base;
	uint64_t* E = B + _Span;
	for (int j = lv.lo; j <= lv.hi; j++) {
		const uint64_t d = 1ull << j;
		const uint64_t* r = R + (_Base >> j);
		if (_Base == 0) nat_asmINtt(d, M, B, r, E);
		else nat_asmINtt2(d, M, B, r, E);
	}
}

// Forward, pre-order: this level's layers over the current chunk, then the next finer
// level, chunk by chunk.  The DRAM level is the whole array, so base stays 0 for it.
inline void ntt_fwd_sched(uint64_t* _Data, uint64_t _Base, const ntt_sched& S, int _S, int _I) {
	if (_S >= S.nlevel)
		return;
	const uint64_t nblk = _S == 0 ? 1 : (1ull << (S.chunk[_S - 1] - S.chunk[_S]));
	for (uint64_t c = 0; c < nblk; c++) {
		const uint64_t cb = _Base + (c << S.chunk[_S]);
		ntt_fwd_level(_Data, cb, 1ull << S.chunk[_S], S.lv[_S], _I);
		ntt_fwd_sched(_Data, cb, S, _S + 1, _I);
	}
}

// Inverse, post-order: the finer levels first, then this level's layers -- the exact
// mirror of ntt_fwd_sched.
inline void ntt_inv_sched(uint64_t* _Data, uint64_t _Base, const ntt_sched& S, int _S, int _I) {
	if (_S >= S.nlevel)
		return;
	const uint64_t nblk = _S == 0 ? 1 : (1ull << (S.chunk[_S - 1] - S.chunk[_S]));
	for (uint64_t c = 0; c < nblk; c++) {
		const uint64_t cb = _Base + (c << S.chunk[_S]);
		ntt_inv_sched(_Data, cb, S, _S + 1, _I);
		ntt_inv_level(_Data, cb, 1ull << S.chunk[_S], S.lv[_S], _I);
	}
}

inline void ntt_workspace::ntt() {
	const ntt_sched S = ntt_sched_for(ntt_scale);
	for (int i = 0; i < 3; i++)
		ntt_fwd_sched(ntt_data[i].data, 0, S, 0, i);
}

inline void ntt_workspace::intt() {
	const ntt_sched S = ntt_sched_for(ntt_scale);
	for (int i = 0; i < 3; i++) {
		ntt_inv_sched(ntt_data[i].data, 0, S, 0, i);
		intt_shr(ntt_scale, mods[i], ntt_data[i].data);
	}
}

// this = a * b in the NTT domain: the fused nat_asmNttMul kernel does "a's last forward
// layer, the pointwise product, b's first inverse layer" in one pass over each modulus
// (both operands must already have been transformed by NTT()).  Out-of-place: this must
// not be the same object as _A or _B.
inline void ntt_workspace::mul(const ntt_workspace& nw_a, const ntt_workspace& nw_b) {
	ntt_scale = nw_a.ntt_scale;
	uint64_t ntt_size = 1ull << ntt_scale;
	for (int i = 0; i < 3; i++) {
		ntt_data[i].resize(nw_a.ntt_data[i].size);
		nat_asmNttMul(ntt_size, ntt_data[i].data, nw_a.ntt_data[i].data, nw_b.ntt_data[i].data, root[i].data, i);
	}
}

// Merge the three moduli into 64-bit words: crt(len) writes the low words into
// ntt_data[0][0..len-1], the two carry words into ntt_data[1][0..1] and returns the middle
// carry word (same contract as the original nat_asmCRT).
inline uint64_t ntt_workspace::crt(uint64_t len) {
	return nat_asmCRT(ntt_data[0].data, ntt_data[1].data, ntt_data[2].data, len);
}

inline void ntt_workspace::save(array_u64& arr, uint64_t len) {
	uint64_t ntt_size = 1ull << ntt_scale;
	if (len - 1 > ntt_size) {
		crt(ntt_size);
		uint64_t cf = add(ntt_data[0].data, ntt_data[0].data, ntt_data[1].data, ntt_data[0].size, 2);
		add(ntt_data[0].data, ntt_data[0].data, &cf, ntt_data[0].size, 1);
	}
	else {
		ntt_data[0].resize(len);
		ntt_data[0][len - 1] = crt(len - 1);
	}
	arr = std::move(ntt_data[0]);
}

}  // namespace nat
