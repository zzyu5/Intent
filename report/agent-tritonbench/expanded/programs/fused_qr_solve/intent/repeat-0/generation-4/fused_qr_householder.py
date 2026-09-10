import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

def _intent_cover_FULL_D1(args):
    bound = int(args["D1"])
    for extent in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, ):
        if extent >= bound:
            return extent
    raise ValueError("no legal full-coverage extent for FULL_D1")

def _intent_cover_FULL_D2(args):
    bound = int(args["D2"])
    for extent in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, ):
        if extent >= bound:
            return extent
    raise ValueError("no legal full-coverage extent for FULL_D2")

def _intent_cover_FULL_D3(args):
    bound = int(args["D3"])
    for extent in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, ):
        if extent >= bound:
            return extent
    raise ValueError("no legal full-coverage extent for FULL_D3")

_intent_tuning_hooks = TuningHooks(("matrix", "rhs", "work_matrix", "work_rhs", ), (False, False, True, True, ))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "D2", "D3", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S2_1", "S3_0", "S3_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.heuristics({
    "FULL_D1": _intent_cover_FULL_D1,
    "FULL_D2": _intent_cover_FULL_D2,
    "FULL_D3": _intent_cover_FULL_D3,
})
@triton.jit
def _intent_kernel(matrix, rhs, work_matrix, work_rhs, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr, FULL_D3: tl.constexpr, FULL_D2: tl.constexpr, FULL_D1: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    v2 = tl.cast(0, tl.float32)
    v3 = tl.cast(2, tl.float32)
    v4 = (0 + tl.arange(0, FULL_D1) * 1)
    v5 = (D1 - 0)
    v6 = (1 - 1)
    v7 = (v5 + v6)
    v8 = ((v7 // 1) - (((v7 % 1) != 0) & (((v7 % 1) < 0) != (1 < 0))))
    v9 = (v8 * 1)
    v10 = (0 + v9)
    v11 = v10
    v12 = (v4 < v11)
    v13 = (0 + tl.arange(0, FULL_D2) * 1)
    v14 = (D2 - 0)
    v15 = (1 - 1)
    v16 = (v14 + v15)
    v17 = ((v16 // 1) - (((v16 % 1) != 0) & (((v16 % 1) < 0) != (1 < 0))))
    v18 = (v17 * 1)
    v19 = (0 + v18)
    v20 = v19
    v21 = (v13 < v20)
    v22 = D1
    v23 = tl.full((FULL_D1,), v22, tl.int64)
    v24 = (v4 < v23)
    v25 = v24[:, None]
    v26 = D2
    v27 = tl.full((FULL_D2,), v26, tl.int64)
    v28 = (v13 < v27)
    v29 = v28[None, :]
    v30 = (v25 & v29)
    v31 = tl.full((FULL_D1, FULL_D2), 0.0, tl.float32)
    v32 = (0 + tl.arange(0, FULL_D1) * 1)
    v33 = (1 - 1)
    v34 = (v5 + v33)
    v35 = ((v34 // 1) - (((v34 % 1) != 0) & (((v34 % 1) < 0) != (1 < 0))))
    v36 = (v35 * 1)
    v37 = (0 + v36)
    v38 = v37
    v39 = (v32 < v38)
    v40 = (0 + tl.arange(0, FULL_D2) * 1)
    v41 = (1 - 1)
    v42 = (v14 + v41)
    v43 = ((v42 // 1) - (((v42 % 1) != 0) & (((v42 % 1) < 0) != (1 < 0))))
    v44 = (v43 * 1)
    v45 = (0 + v44)
    v46 = v45
    v47 = (v40 < v46)
    v48 = D1
    v49 = tl.full((FULL_D1,), v48, tl.int64)
    v50 = (v32 < v49)
    v51 = v50[:, None]
    v52 = D2
    v53 = tl.full((FULL_D2,), v52, tl.int64)
    v54 = (v40 < v53)
    v55 = v54[None, :]
    v56 = (v51 & v55)
    v57 = tl.reshape(v12, (FULL_D1, 1), can_reorder=False)
    v58 = tl.broadcast_to(v57, (FULL_D1, FULL_D2, ))
    v59 = (v30 & v58)
    v60 = tl.reshape(v21, (1, FULL_D2), can_reorder=False)
    v61 = tl.broadcast_to(v60, (FULL_D1, FULL_D2, ))
    v62 = (v59 & v61)
    v63 = tl.reshape(v39, (FULL_D1, 1), can_reorder=False)
    v64 = tl.broadcast_to(v63, (FULL_D1, FULL_D2, ))
    v65 = (v56 & v64)
    v66 = tl.reshape(v47, (1, FULL_D2), can_reorder=False)
    v67 = tl.broadcast_to(v66, (FULL_D1, FULL_D2, ))
    v68 = (v65 & v67)
    v69 = tl.load(tl.make_block_ptr(base=(matrix + tl.cast(0, tl.int64) * S0_0 + tl.cast(0, tl.int64) * S0_1), shape=((D1 - tl.cast(0, tl.int64)), (D2 - tl.cast(0, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FULL_D1, FULL_D2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
    tl.store(tl.make_block_ptr(base=(work_matrix + tl.cast(0, tl.int64) * S2_0 + tl.cast(0, tl.int64) * S2_1), shape=((D1 - tl.cast(0, tl.int64)), (D2 - tl.cast(0, tl.int64))), strides=(S2_0, S2_1), offsets=(0, 0), block_shape=(FULL_D1, FULL_D2), order=(1, 0)), tl.cast(v69, tl.float32), boundary_check=(0, 1))
    v70 = (0 + tl.arange(0, FULL_D1) * 1)
    v71 = (1 - 1)
    v72 = (v5 + v71)
    v73 = ((v72 // 1) - (((v72 % 1) != 0) & (((v72 % 1) < 0) != (1 < 0))))
    v74 = (v73 * 1)
    v75 = (0 + v74)
    v76 = v75
    v77 = (v70 < v76)
    v78 = (0 + tl.arange(0, FULL_D3) * 1)
    v79 = (D3 - 0)
    v80 = (1 - 1)
    v81 = (v79 + v80)
    v82 = ((v81 // 1) - (((v81 % 1) != 0) & (((v81 % 1) < 0) != (1 < 0))))
    v83 = (v82 * 1)
    v84 = (0 + v83)
    v85 = v84
    v86 = (v78 < v85)
    v87 = D1
    v88 = tl.full((FULL_D1,), v87, tl.int64)
    v89 = (v70 < v88)
    v90 = v89[:, None]
    v91 = D3
    v92 = tl.full((FULL_D3,), v91, tl.int64)
    v93 = (v78 < v92)
    v94 = v93[None, :]
    v95 = (v90 & v94)
    v96 = tl.full((FULL_D1, FULL_D3), 0.0, tl.float32)
    v97 = (0 + tl.arange(0, FULL_D1) * 1)
    v98 = (1 - 1)
    v99 = (v5 + v98)
    v100 = ((v99 // 1) - (((v99 % 1) != 0) & (((v99 % 1) < 0) != (1 < 0))))
    v101 = (v100 * 1)
    v102 = (0 + v101)
    v103 = v102
    v104 = (v97 < v103)
    v105 = (0 + tl.arange(0, FULL_D3) * 1)
    v106 = (1 - 1)
    v107 = (v79 + v106)
    v108 = ((v107 // 1) - (((v107 % 1) != 0) & (((v107 % 1) < 0) != (1 < 0))))
    v109 = (v108 * 1)
    v110 = (0 + v109)
    v111 = v110
    v112 = (v105 < v111)
    v113 = D1
    v114 = tl.full((FULL_D1,), v113, tl.int64)
    v115 = (v97 < v114)
    v116 = v115[:, None]
    v117 = D3
    v118 = tl.full((FULL_D3,), v117, tl.int64)
    v119 = (v105 < v118)
    v120 = v119[None, :]
    v121 = (v116 & v120)
    v122 = tl.reshape(v77, (FULL_D1, 1), can_reorder=False)
    v123 = tl.broadcast_to(v122, (FULL_D1, FULL_D3, ))
    v124 = (v95 & v123)
    v125 = tl.reshape(v86, (1, FULL_D3), can_reorder=False)
    v126 = tl.broadcast_to(v125, (FULL_D1, FULL_D3, ))
    v127 = (v124 & v126)
    v128 = tl.reshape(v104, (FULL_D1, 1), can_reorder=False)
    v129 = tl.broadcast_to(v128, (FULL_D1, FULL_D3, ))
    v130 = (v121 & v129)
    v131 = tl.reshape(v112, (1, FULL_D3), can_reorder=False)
    v132 = tl.broadcast_to(v131, (FULL_D1, FULL_D3, ))
    v133 = (v130 & v132)
    v134 = tl.load(tl.make_block_ptr(base=(rhs + tl.cast(0, tl.int64) * S1_0 + tl.cast(0, tl.int64) * S1_1), shape=((D1 - tl.cast(0, tl.int64)), (D3 - tl.cast(0, tl.int64))), strides=(S1_0, S1_1), offsets=(0, 0), block_shape=(FULL_D1, FULL_D3), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
    tl.store(tl.make_block_ptr(base=(work_rhs + tl.cast(0, tl.int64) * S3_0 + tl.cast(0, tl.int64) * S3_1), shape=((D1 - tl.cast(0, tl.int64)), (D3 - tl.cast(0, tl.int64))), strides=(S3_0, S3_1), offsets=(0, 0), block_shape=(FULL_D1, FULL_D3), order=(1, 0)), tl.cast(v134, tl.float32), boundary_check=(0, 1))
    for iv135 in range(0, D2, 1):
        v136 = v2
        for iv137 in range(iv135, D1, 1):
            v138 = D1
            v139 = (iv137 < v138)
            v140 = (iv137 >= 0)
            v141 = (v140 & v139)
            v142 = D2
            v143 = (iv135 < v142)
            v144 = (iv135 >= 0)
            v145 = (v144 & v143)
            v146 = (v141 & v145)
            v147 = tl.load((work_matrix + (iv137) * S2_0 + (iv135) * S2_1), mask=v146, other=0.0)
            v148 = (v147 * v147)
            v149 = (v136 + v148)
            v136 = v149
        v150 = (v136 > v2)
        if v150:
            v151 = tl.cast(1, tl.float32)
            v152 = tl.cast(1, tl.float32)
            v153 = tl.cast(2, tl.float32)
            v154 = tl.fdiv(tl.cast(v152, tl.float32), tl.cast(v153, tl.float32), ieee_rounding=True)
            v155 = (v136 > v151)
            if v155:
                v156 = v136
            else:
                v156 = v151
            v157 = v156
            for iv158 in range(0, 16, 1):
                v159 = tl.fdiv(tl.cast(v136, tl.float32), tl.cast(v157, tl.float32), ieee_rounding=True)
                v160 = (v157 + v159)
                v161 = (v154 * v160)
                v157 = v161
            v162 = D1
            v163 = (iv135 < v162)
            v164 = (iv135 >= 0)
            v165 = (v164 & v163)
            v166 = D2
            v167 = (iv135 < v166)
            v168 = (iv135 >= 0)
            v169 = (v168 & v167)
            v170 = (v165 & v169)
            v171 = tl.load((work_matrix + (iv135) * S2_0 + (iv135) * S2_1), mask=v170, other=0.0)
            v172 = (v171 >= v2)
            if v172:
                v174 = (-v157)
                v173 = v174
            else:
                v173 = v157
            v175 = (v171 - v173)
            v176 = (v175 * v175)
            v177 = (iv135 + 1)
            v178 = v176
            for iv179 in range(v177, D1, 1):
                v180 = D1
                v181 = (iv179 < v180)
                v182 = (iv179 >= 0)
                v183 = (v182 & v181)
                v184 = D2
                v185 = (iv135 < v184)
                v186 = (iv135 >= 0)
                v187 = (v186 & v185)
                v188 = (v183 & v187)
                v189 = tl.load((work_matrix + (iv179) * S2_0 + (iv135) * S2_1), mask=v188, other=0.0)
                v190 = (v189 * v189)
                v191 = (v178 + v190)
                v178 = v191
            v192 = (iv135 + 1)
            for iv193 in range(v192, D2, 1):
                v194 = D1
                v195 = (iv135 < v194)
                v196 = (iv135 >= 0)
                v197 = (v196 & v195)
                v198 = D2
                v199 = (iv193 < v198)
                v200 = (iv193 >= 0)
                v201 = (v200 & v199)
                v202 = (v197 & v201)
                v203 = tl.load((work_matrix + (iv135) * S2_0 + (iv193) * S2_1), mask=v202, other=0.0)
                v204 = (v175 * v203)
                v205 = (iv135 + 1)
                v206 = v204
                for iv207 in range(v205, D1, 1):
                    v208 = D1
                    v209 = (iv207 < v208)
                    v210 = (iv207 >= 0)
                    v211 = (v210 & v209)
                    v212 = D2
                    v213 = (iv135 < v212)
                    v214 = (iv135 >= 0)
                    v215 = (v214 & v213)
                    v216 = (v211 & v215)
                    v217 = tl.load((work_matrix + (iv207) * S2_0 + (iv135) * S2_1), mask=v216, other=0.0)
                    v218 = (iv207 >= 0)
                    v219 = (v218 & v209)
                    v220 = (iv193 < v212)
                    v221 = (iv193 >= 0)
                    v222 = (v221 & v220)
                    v223 = (v219 & v222)
                    v224 = tl.load((work_matrix + (iv207) * S2_0 + (iv193) * S2_1), mask=v223, other=0.0)
                    v225 = (v217 * v224)
                    v226 = (v206 + v225)
                    v206 = v226
                v227 = (v3 * v206)
                v228 = tl.fdiv(tl.cast(v227, tl.float32), tl.cast(v178, tl.float32), ieee_rounding=True)
                v229 = (iv135 >= 0)
                v230 = (v229 & v195)
                v231 = (iv193 >= 0)
                v232 = (v231 & v199)
                v233 = (v230 & v232)
                v234 = tl.load((work_matrix + (iv135) * S2_0 + (iv193) * S2_1), mask=v233, other=0.0)
                v235 = (v228 * v175)
                v236 = (v234 - v235)
                v237 = (iv135 >= 0)
                v238 = (v237 & v195)
                v239 = (iv193 >= 0)
                v240 = (v239 & v199)
                v241 = (v238 & v240)
                tl.store((work_matrix + (iv135) * S2_0 + (iv193) * S2_1), v236, mask=v241)
                v242 = (iv135 + 1)
                for iv243 in range(v242, D1, 1):
                    v244 = D1
                    v245 = (iv243 < v244)
                    v246 = (iv243 >= 0)
                    v247 = (v246 & v245)
                    v248 = D2
                    v249 = (iv193 < v248)
                    v250 = (iv193 >= 0)
                    v251 = (v250 & v249)
                    v252 = (v247 & v251)
                    v253 = tl.load((work_matrix + (iv243) * S2_0 + (iv193) * S2_1), mask=v252, other=0.0)
                    v254 = (iv243 >= 0)
                    v255 = (v254 & v245)
                    v256 = (iv135 < v248)
                    v257 = (iv135 >= 0)
                    v258 = (v257 & v256)
                    v259 = (v255 & v258)
                    v260 = tl.load((work_matrix + (iv243) * S2_0 + (iv135) * S2_1), mask=v259, other=0.0)
                    v261 = (v228 * v260)
                    v262 = (v253 - v261)
                    v263 = (iv243 >= 0)
                    v264 = (v263 & v245)
                    v265 = (iv193 >= 0)
                    v266 = (v265 & v249)
                    v267 = (v264 & v266)
                    tl.store((work_matrix + (iv243) * S2_0 + (iv193) * S2_1), v262, mask=v267)
            for iv268 in range(0, D3, 1):
                v269 = D1
                v270 = (iv135 < v269)
                v271 = (iv135 >= 0)
                v272 = (v271 & v270)
                v273 = D3
                v274 = (iv268 < v273)
                v275 = (iv268 >= 0)
                v276 = (v275 & v274)
                v277 = (v272 & v276)
                v278 = tl.load((work_rhs + (iv135) * S3_0 + (iv268) * S3_1), mask=v277, other=0.0)
                v279 = (v175 * v278)
                v280 = (iv135 + 1)
                v281 = v279
                for iv282 in range(v280, D1, 1):
                    v283 = D1
                    v284 = (iv282 < v283)
                    v285 = (iv282 >= 0)
                    v286 = (v285 & v284)
                    v287 = D2
                    v288 = (iv135 < v287)
                    v289 = (iv135 >= 0)
                    v290 = (v289 & v288)
                    v291 = (v286 & v290)
                    v292 = tl.load((work_matrix + (iv282) * S2_0 + (iv135) * S2_1), mask=v291, other=0.0)
                    v293 = D1
                    v294 = (iv282 < v293)
                    v295 = (iv282 >= 0)
                    v296 = (v295 & v294)
                    v297 = D3
                    v298 = (iv268 < v297)
                    v299 = (iv268 >= 0)
                    v300 = (v299 & v298)
                    v301 = (v296 & v300)
                    v302 = tl.load((work_rhs + (iv282) * S3_0 + (iv268) * S3_1), mask=v301, other=0.0)
                    v303 = (v292 * v302)
                    v304 = (v281 + v303)
                    v281 = v304
                v305 = (v3 * v281)
                v306 = tl.fdiv(tl.cast(v305, tl.float32), tl.cast(v178, tl.float32), ieee_rounding=True)
                v307 = (iv135 >= 0)
                v308 = (v307 & v270)
                v309 = (iv268 >= 0)
                v310 = (v309 & v274)
                v311 = (v308 & v310)
                v312 = tl.load((work_rhs + (iv135) * S3_0 + (iv268) * S3_1), mask=v311, other=0.0)
                v313 = (v306 * v175)
                v314 = (v312 - v313)
                v315 = (iv135 >= 0)
                v316 = (v315 & v270)
                v317 = (iv268 >= 0)
                v318 = (v317 & v274)
                v319 = (v316 & v318)
                tl.store((work_rhs + (iv135) * S3_0 + (iv268) * S3_1), v314, mask=v319)
                v320 = (iv135 + 1)
                for iv321 in range(v320, D1, 1):
                    v322 = D1
                    v323 = (iv321 < v322)
                    v324 = (iv321 >= 0)
                    v325 = (v324 & v323)
                    v326 = D3
                    v327 = (iv268 < v326)
                    v328 = (iv268 >= 0)
                    v329 = (v328 & v327)
                    v330 = (v325 & v329)
                    v331 = tl.load((work_rhs + (iv321) * S3_0 + (iv268) * S3_1), mask=v330, other=0.0)
                    v332 = D1
                    v333 = (iv321 < v332)
                    v334 = (iv321 >= 0)
                    v335 = (v334 & v333)
                    v336 = D2
                    v337 = (iv135 < v336)
                    v338 = (iv135 >= 0)
                    v339 = (v338 & v337)
                    v340 = (v335 & v339)
                    v341 = tl.load((work_matrix + (iv321) * S2_0 + (iv135) * S2_1), mask=v340, other=0.0)
                    v342 = (v306 * v341)
                    v343 = (v331 - v342)
                    v344 = (iv321 >= 0)
                    v345 = (v344 & v323)
                    v346 = (iv268 >= 0)
                    v347 = (v346 & v327)
                    v348 = (v345 & v347)
                    tl.store((work_rhs + (iv321) * S3_0 + (iv268) * S3_1), v343, mask=v348)
            v349 = (iv135 >= 0)
            v350 = (v349 & v163)
            v351 = (iv135 >= 0)
            v352 = (v351 & v167)
            v353 = (v350 & v352)
            tl.store((work_matrix + (iv135) * S2_0 + (iv135) * S2_1), v173, mask=v353)
            v354 = (iv135 + 1)
            for iv355 in range(v354, D1, 1):
                v356 = D1
                v357 = (iv355 < v356)
                v358 = (iv355 >= 0)
                v359 = (v358 & v357)
                v360 = D2
                v361 = (iv135 < v360)
                v362 = (iv135 >= 0)
                v363 = (v362 & v361)
                v364 = (v359 & v363)
                tl.store((work_matrix + (iv355) * S2_0 + (iv135) * S2_1), v2, mask=v364)
        else:
            v365 = D1
            v366 = (iv135 < v365)
            v367 = (iv135 >= 0)
            v368 = (v367 & v366)
            v369 = D2
            v370 = (iv135 < v369)
            v371 = (iv135 >= 0)
            v372 = (v371 & v370)
            v373 = (v368 & v372)
            tl.store((work_matrix + (iv135) * S2_0 + (iv135) * S2_1), v2, mask=v373)
            v374 = (iv135 + 1)
            for iv375 in range(v374, D1, 1):
                v376 = D1
                v377 = (iv375 < v376)
                v378 = (iv375 >= 0)
                v379 = (v378 & v377)
                v380 = D2
                v381 = (iv135 < v380)
                v382 = (iv135 >= 0)
                v383 = (v382 & v381)
                v384 = (v379 & v383)
                tl.store((work_matrix + (iv375) * S2_0 + (iv135) * S2_1), v2, mask=v384)

def launch(matrix, rhs, work_matrix, work_rhs):
    D1 = matrix.shape[0]
    D2 = matrix.shape[1]
    D3 = rhs.shape[1]
    S0_0 = matrix.stride(0)
    S0_1 = matrix.stride(1)
    S1_0 = rhs.stride(0)
    S1_1 = rhs.stride(1)
    S2_0 = work_matrix.stride(0)
    S2_1 = work_matrix.stride(1)
    S3_0 = work_rhs.stride(0)
    S3_1 = work_rhs.stride(1)
    grid = lambda META: (1,)
    return _intent_kernel[grid](matrix, rhs, work_matrix, work_rhs, D1, D2, D3, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1, S3_0, S3_1)

def run(matrix, rhs, work_matrix, work_rhs):
    launch(matrix, rhs, work_matrix, work_rhs)
    return None
