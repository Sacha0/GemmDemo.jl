using Unicode
using SIMD

# macro to generate simple three-nested-loop implementations
macro def_gemm_xyz!(xyz)
    x, y, z = Symbol.(graphemes(String(xyz)))

    indexrange(q) =
        q == :i ? :(1:size(C, 1)) :
        q == :j ? :(1:size(C, 2)) :
        q == :p ? :(1:size(A, 2)) :
        throw(ArgumentError("$q"))

    return quote
        function $(esc(Symbol(:gemm_, xyz, :!)))(C, A, B)
            for $(x) in $(indexrange(x)),
                 $(y) in $(indexrange(y)),
                  $(z) in $(indexrange(z))
                @inbounds C[i, j] += A[i, p] * B[p, j]
            end
            return C
        end
    end
end
@def_gemm_xyz! jip # defines gemm_jip!(C, A, B)

# non-packing gemm implementation
function gemm_nonpacking!(C, A, B, (cacheM, cacheK, cacheN), ::Val{microM}, ::Val{microN}) where {microM,microN}
    m = size(C, 1)
    k = size(A, 2)
    n = size(C, 2)
    # cache-blocking loops
    for cachejstart in 1:cacheN:n; cachejend = min(cachejstart + cacheN - 1, n)
        for cachepstart in 1:cacheK:k; cachepend = min(cachepstart + cacheK - 1, k)
            for cacheistart in 1:cacheM:m; cacheiend = min(cacheistart + cacheM - 1, m)
                # macrokernel loops
                for macrojstart in cachejstart:microN:cachejend
                    for macroistart in cacheistart:microM:cacheiend
                        gemm_nonpacking_microkernel!(C, A, B,
                            macroistart, macrojstart,
                            cachepstart, cachepend,
                            Val(microM), Val(microN))
                    end
                end
            end
        end
    end
    return C
end
@inline function gemm_nonpacking_microkernel!(C, A, B, i, j, pstart, pend, ::Val{12}, ::Val{4})
    T = eltype(C)
    m = size(C, 1)
    
    # load 12x4 block Cij into vector registers
    ptrCij = pointer(C, (j - 1)*m + i) # pointer to C[i, j]
    # load Cij[1:12, 1]
    Cij_r1c1vec = vload(Vec{4,T}, ptrCij              ) # C[i  , j]
    Cij_r2c1vec = vload(Vec{4,T}, ptrCij + 4*sizeof(T)) # C[i+4, j]
    Cij_r3c1vec = vload(Vec{4,T}, ptrCij + 8*sizeof(T)) # C[i+8, j]
    # load Cij[1:12, 2]
    Cij_r1c2vec = vload(Vec{4,T}, ptrCij +       m*sizeof(T)) # C[i  , j+1]
    Cij_r2c2vec = vload(Vec{4,T}, ptrCij + (4 + m)*sizeof(T)) # C[i+4, j+1]
    Cij_r3c2vec = vload(Vec{4,T}, ptrCij + (8 + m)*sizeof(T)) # C[i+8, j+1]
    # load Cij[1:12, 3]
    Cij_r1c3vec = vload(Vec{4,T}, ptrCij +       2m*sizeof(T)) # C[i  , j+2]
    Cij_r2c3vec = vload(Vec{4,T}, ptrCij + (4 + 2m)*sizeof(T)) # C[i+4, j+2]
    Cij_r3c3vec = vload(Vec{4,T}, ptrCij + (8 + 2m)*sizeof(T)) # C[i+8, j+2]
    # load Cij[1:12, 4]
    Cij_r1c4vec = vload(Vec{4,T}, ptrCij +       3m*sizeof(T)) # C[i  , j+3]
    Cij_r2c4vec = vload(Vec{4,T}, ptrCij + (4 + 3m)*sizeof(T)) # C[i+4, j+3]
    Cij_r3c4vec = vload(Vec{4,T}, ptrCij + (8 + 3m)*sizeof(T)) # C[i+8, j+3]
    
    # update Cij with series of outer products
    # from the associated Aip and Bpj micropanels
    for p in pstart:pend
        # update with A[i:(i+11), p] * B[p, j:(j+3)] outer product
        ptrAip = pointer(A, (p - 1)*m + i) # pointer to A[i, p]
        Aip_vec1 = vload(Vec{4,T}, ptrAip)
        Aip_vec2 = vload(Vec{4,T}, ptrAip + 4*sizeof(T))
        Aip_vec3 = vload(Vec{4,T}, ptrAip + 8*sizeof(T))
        # Cij[1:12, 1] += A[i:(i+11), p] * B[p, j]
        Bpj_c1vec = Vec{4,T}(@inbounds B[p, j])
        Cij_r1c1vec = fma(Aip_vec1, Bpj_c1vec, Cij_r1c1vec)
        Cij_r2c1vec = fma(Aip_vec2, Bpj_c1vec, Cij_r2c1vec)
        Cij_r3c1vec = fma(Aip_vec3, Bpj_c1vec, Cij_r3c1vec)
        # Cij[1:12, 2] += A[i:(i+11), p] * B[p, j+1]
        Bpj_c2vec = Vec{4,T}(@inbounds B[p, j+1])
        Cij_r1c2vec = fma(Aip_vec1, Bpj_c2vec, Cij_r1c2vec)
        Cij_r2c2vec = fma(Aip_vec2, Bpj_c2vec, Cij_r2c2vec)
        Cij_r3c2vec = fma(Aip_vec3, Bpj_c2vec, Cij_r3c2vec)
        # Cij[1:12, 3] += A[i:(i+11), p] * B[p, j+2]
        Bpj_c3vec = Vec{4,T}(@inbounds B[p, j+2])
        Cij_r1c3vec = fma(Aip_vec1, Bpj_c3vec, Cij_r1c3vec)
        Cij_r2c3vec = fma(Aip_vec2, Bpj_c3vec, Cij_r2c3vec)
        Cij_r3c3vec = fma(Aip_vec3, Bpj_c3vec, Cij_r3c3vec)
        # Cij[1:12, 4] += A[i:(i+11), p] * B[p, j+3]
        Bpj_c4vec = Vec{4,T}(@inbounds B[p, j+3])
        Cij_r1c4vec = fma(Aip_vec1, Bpj_c4vec, Cij_r1c4vec)
        Cij_r2c4vec = fma(Aip_vec2, Bpj_c4vec, Cij_r2c4vec)
        Cij_r3c4vec = fma(Aip_vec3, Bpj_c4vec, Cij_r3c4vec)
    end

    # store Cij[1:12, 1]
    vstore(Cij_r1c1vec, ptrCij              ) # C[i  , j]
    vstore(Cij_r2c1vec, ptrCij + 4*sizeof(T)) # C[i+4, j]
    vstore(Cij_r3c1vec, ptrCij + 8*sizeof(T)) # C[i+8, j]
    # store Cij[1:12, 2]
    vstore(Cij_r1c2vec, ptrCij +       m*sizeof(T)) # C[i  , j+1]
    vstore(Cij_r2c2vec, ptrCij + (4 + m)*sizeof(T)) # C[i+4, j+1]
    vstore(Cij_r3c2vec, ptrCij + (8 + m)*sizeof(T)) # C[i+8, j+1]
    # store Cij[1:12, 3]
    vstore(Cij_r1c3vec, ptrCij +       2m*sizeof(T)) # C[i  , j+2]
    vstore(Cij_r2c3vec, ptrCij + (4 + 2m)*sizeof(T)) # C[i+4, j+2]
    vstore(Cij_r3c3vec, ptrCij + (8 + 2m)*sizeof(T)) # C[i+8, j+2]
    # store Cij[1:12, 4]
    vstore(Cij_r1c4vec, ptrCij +       3m*sizeof(T)) # C[i  , j+3]
    vstore(Cij_r2c4vec, ptrCij + (4 + 3m)*sizeof(T)) # C[i+4, j+3]
    vstore(Cij_r3c4vec, ptrCij + (8 + 3m)*sizeof(T)) # C[i+8, j+3]
    return nothing
end

# packing gemm implementation
function gemm_packing!(C, A, B, Abuff, Bbuff, (cacheM, cacheK, cacheN), ::Val{microM}, ::Val{microN}) where {microM,microN}
    m = size(C, 1)
    k = size(A, 2)
    n = size(C, 2)
    # cache-blocking loops
    for cachejstart in 1:cacheN:n; cachejend = min(cachejstart + cacheN - 1, n)
        for cachepstart in 1:cacheK:k; cachepend = min(cachepstart + cacheK - 1, k)
            packBbuffer!(Bbuff, B, cachepstart, cachepend, cachejstart, cachejend, microN)
            for cacheistart in 1:cacheM:m; cacheiend = min(cacheistart + cacheM - 1, m)
                packAbuffer!(Abuff, A, cacheistart, cacheiend, cachepstart, cachepend, microM)
                # macrokernel loops
                for macrojstart in cachejstart:microN:cachejend
                    for macroistart in cacheistart:microM:cacheiend
                        gemm_packing_microkernel!(C, Abuff, Bbuff,
                            macroistart, cacheistart,
                            macrojstart, cachejstart,
                            cachepstart, cachepend,
                            Val(microM), Val(microN))
                    end
                end
            end
        end
    end
    return C
end
@inline function gemm_packing_microkernel!(C, Abuff, Bbuff,
                                    istart, blockistart,
                                    jstart, blockjstart,
                                    blockpstart, blockpend,
                                    ::Val{12}, ::Val{4})
    microM, microN = 12, 4
    T = eltype(C)
    m = size(C, 1)

    # load 12x4 block Cij into vector registers
    Cstart = (jstart - 1)*m + istart
    ptrCij = pointer(C, Cstart) # pointer to C[i, j]
    # load Cij[1:12, 1]
    Cij_r1c1vec = vload(Vec{4,T}, ptrCij              ) # C[i  , j]
    Cij_r2c1vec = vload(Vec{4,T}, ptrCij + 4*sizeof(T)) # C[i+4, j]
    Cij_r3c1vec = vload(Vec{4,T}, ptrCij + 8*sizeof(T)) # C[i+8, j]
    # load Cij[1:12, 2]
    Cij_r1c2vec = vload(Vec{4,T}, ptrCij +       m*sizeof(T)) # C[i  , j+1]
    Cij_r2c2vec = vload(Vec{4,T}, ptrCij + (4 + m)*sizeof(T)) # C[i+4, j+1]
    Cij_r3c2vec = vload(Vec{4,T}, ptrCij + (8 + m)*sizeof(T)) # C[i+8, j+1]
    # load Cij[:, 3]
    Cij_r1c3vec = vload(Vec{4,T}, ptrCij +       2m*sizeof(T)) # C[i  , j+2]
    Cij_r2c3vec = vload(Vec{4,T}, ptrCij + (4 + 2m)*sizeof(T)) # C[i+4, j+2]
    Cij_r3c3vec = vload(Vec{4,T}, ptrCij + (8 + 2m)*sizeof(T)) # C[i+8, j+2]
    # load Cij[:, 4]
    Cij_r1c4vec = vload(Vec{4,T}, ptrCij +       3m*sizeof(T)) # C[i  , j+3]
    Cij_r2c4vec = vload(Vec{4,T}, ptrCij + (4 + 3m)*sizeof(T)) # C[i+4, j+3]
    Cij_r3c4vec = vload(Vec{4,T}, ptrCij + (8 + 3m)*sizeof(T)) # C[i+8, j+3]
    
    blockplength = blockpend - blockpstart + 1
    Abuffstart = (istart - blockistart)*blockplength + 1
    Bbuffstart = (jstart - blockjstart)*blockplength + 1
    ptrAbuffpos = pointer(Abuff, Abuffstart)
    indBbuffpos = Bbuffstart

    # update Cij with series of outer products
    # from the associated Aip and Bpj micropanels
    for p in blockpstart:blockpend
        # update with A[i:(i+11), p] * B[p, j:(j+3)] outer product
        # ptrAip = pointer(A, (p - 1)*m + i) # pointer to A[i, p]
        Aip_vec1 = vload(Vec{4,T}, ptrAbuffpos)
        Aip_vec2 = vload(Vec{4,T}, ptrAbuffpos + 4*sizeof(T))
        Aip_vec3 = vload(Vec{4,T}, ptrAbuffpos + 8*sizeof(T))
        # Cij[1:12, 1] += A[i:(i+11), p] * B[p, j]
        Bpj_c1vec = Vec{4,T}(@inbounds Bbuff[indBbuffpos])
        Cij_r1c1vec = fma(Aip_vec1, Bpj_c1vec, Cij_r1c1vec)
        Cij_r2c1vec = fma(Aip_vec2, Bpj_c1vec, Cij_r2c1vec)
        Cij_r3c1vec = fma(Aip_vec3, Bpj_c1vec, Cij_r3c1vec)
        # Cij[1:12, 2] += A[i:(i+11), p] * B[p, j+1]
        Bpj_c2vec = Vec{4,T}(@inbounds Bbuff[indBbuffpos + 1])
        Cij_r1c2vec = fma(Aip_vec1, Bpj_c2vec, Cij_r1c2vec)
        Cij_r2c2vec = fma(Aip_vec2, Bpj_c2vec, Cij_r2c2vec)
        Cij_r3c2vec = fma(Aip_vec3, Bpj_c2vec, Cij_r3c2vec)
        # Cij[1:12, 3] += A[i:(i+11), p] * B[p, j+2]
        Bpj_c3vec = Vec{4,T}(@inbounds Bbuff[indBbuffpos + 2])
        Cij_r1c3vec = fma(Aip_vec1, Bpj_c3vec, Cij_r1c3vec)
        Cij_r2c3vec = fma(Aip_vec2, Bpj_c3vec, Cij_r2c3vec)
        Cij_r3c3vec = fma(Aip_vec3, Bpj_c3vec, Cij_r3c3vec)
        # Cij[1:12, 4] += A[i:(i+11), p] * B[p, j+3]
        Bpj_c4vec = Vec{4,T}(@inbounds Bbuff[indBbuffpos + 3])
        Cij_r1c4vec = fma(Aip_vec1, Bpj_c4vec, Cij_r1c4vec)
        Cij_r2c4vec = fma(Aip_vec2, Bpj_c4vec, Cij_r2c4vec)
        Cij_r3c4vec = fma(Aip_vec3, Bpj_c4vec, Cij_r3c4vec)
        ptrAbuffpos += microM*sizeof(T)
        indBbuffpos += microN
    end

    # store Cij[:, 1]
    vstore(Cij_r1c1vec, ptrCij              ) # C[i  , j]
    vstore(Cij_r2c1vec, ptrCij + 4*sizeof(T)) # C[i+4, j]
    vstore(Cij_r3c1vec, ptrCij + 8*sizeof(T)) # C[i+8, j]
    # store Cij[:, 2]
    vstore(Cij_r1c2vec, ptrCij +       m*sizeof(T)) # C[i  , j+1]
    vstore(Cij_r2c2vec, ptrCij + (4 + m)*sizeof(T)) # C[i+4, j+1]
    vstore(Cij_r3c2vec, ptrCij + (8 + m)*sizeof(T)) # C[i+8, j+1]
    # store Cij[:, 3]
    vstore(Cij_r1c3vec, ptrCij +       2m*sizeof(T)) # C[i  , j+2]
    vstore(Cij_r2c3vec, ptrCij + (4 + 2m)*sizeof(T)) # C[i+4, j+2]
    vstore(Cij_r3c3vec, ptrCij + (8 + 2m)*sizeof(T)) # C[i+8, j+2]
    # store Cij[:, 4]
    vstore(Cij_r1c4vec, ptrCij +       3m*sizeof(T)) # C[i  , j+3]
    vstore(Cij_r2c4vec, ptrCij + (4 + 3m)*sizeof(T)) # C[i+4, j+3]
    vstore(Cij_r3c4vec, ptrCij + (8 + 3m)*sizeof(T)) # C[i+8, j+3]
    return nothing
end
@inline function gemm_packing_microkernel!(C, Abuff, Bbuff,
                                    istart, blockistart,
                                    jstart, blockjstart,
                                    blockpstart, blockpend,
                                    ::Val{8}, ::Val{6})
    microM, microN = 8, 6
    T = eltype(C)
    m = size(C, 1)
    
    # load 8x6 block Cij into vector registers
    Cstart = (jstart - 1)*m + istart
    ptrCij = pointer(C, Cstart) # pointer to C[i, j]
    # load Cij[:, 1]
    Cij_r1c1vec = vload(Vec{4,T}, ptrCij              ) # C[i  , j]
    Cij_r2c1vec = vload(Vec{4,T}, ptrCij + 4*sizeof(T)) # C[i+4, j]
    # load Cij[:, 2]
    Cij_r1c2vec = vload(Vec{4,T}, ptrCij +       m*sizeof(T)) # C[i  , j+1]
    Cij_r2c2vec = vload(Vec{4,T}, ptrCij + (4 + m)*sizeof(T)) # C[i+4, j+1]
    # load Cij[:, 3]
    Cij_r1c3vec = vload(Vec{4,T}, ptrCij +       2m*sizeof(T)) # C[i  , j+2]
    Cij_r2c3vec = vload(Vec{4,T}, ptrCij + (4 + 2m)*sizeof(T)) # C[i+4, j+2]
    # load Cij[:, 4]
    Cij_r1c4vec = vload(Vec{4,T}, ptrCij +       3m*sizeof(T)) # C[i  , j+3]
    Cij_r2c4vec = vload(Vec{4,T}, ptrCij + (4 + 3m)*sizeof(T)) # C[i+4, j+3]
    # load Cij[:, 5]
    Cij_r1c5vec = vload(Vec{4,T}, ptrCij +       4m*sizeof(T)) # C[i  , j+4]
    Cij_r2c5vec = vload(Vec{4,T}, ptrCij + (4 + 4m)*sizeof(T)) # C[i+4, j+4]
    # load Cij[:, 6]
    Cij_r1c6vec = vload(Vec{4,T}, ptrCij +       5m*sizeof(T)) # C[i  , j+5]
    Cij_r2c6vec = vload(Vec{4,T}, ptrCij + (4 + 5m)*sizeof(T)) # C[i+4, j+6]    

    blockplength = blockpend - blockpstart + 1
    Abuffstart = (istart - blockistart)*blockplength + 1
    Bbuffstart = (jstart - blockjstart)*blockplength + 1
    ptrAbuffpos = pointer(Abuff, Abuffstart)
    indBbuffpos = Bbuffstart
    
    # update Cij with series of outer products
    # from the associated Aip and Bpj micropanels
    for p in blockpstart:blockpend
        # update with A[i:(i+7), p] * B[p, j:(j+5)] outer product
        # ptrAip = pointer(A, (p - 1)*m + i) # pointer to A[i, p]
        Aip_vec1 = vload(Vec{4,T}, ptrAbuffpos)
        Aip_vec2 = vload(Vec{4,T}, ptrAbuffpos + 4*sizeof(T))
        # Cij[1:8, 1] += A[i:(i+7), p] * B[p, j]
        Bpj_c1vec = Vec{4,T}(@inbounds Bbuff[indBbuffpos])
        Cij_r1c1vec = fma(Aip_vec1, Bpj_c1vec, Cij_r1c1vec)
        Cij_r2c1vec = fma(Aip_vec2, Bpj_c1vec, Cij_r2c1vec)
        # Cij[1:8, 2] += A[i:(i+7), p] * B[p, j+1]
        Bpj_c2vec = Vec{4,T}(@inbounds Bbuff[indBbuffpos + 1])
        Cij_r1c2vec = fma(Aip_vec1, Bpj_c2vec, Cij_r1c2vec)
        Cij_r2c2vec = fma(Aip_vec2, Bpj_c2vec, Cij_r2c2vec)
        # Cij[1:8, 3] += A[i:(i+7), p] * B[p, j+2]
        Bpj_c3vec = Vec{4,T}(@inbounds Bbuff[indBbuffpos + 2])
        Cij_r1c3vec = fma(Aip_vec1, Bpj_c3vec, Cij_r1c3vec)
        Cij_r2c3vec = fma(Aip_vec2, Bpj_c3vec, Cij_r2c3vec)
        # Cij[1:8, 4] += A[i:(i+7), p] * B[p, j+3]
        Bpj_c4vec = Vec{4,T}(@inbounds Bbuff[indBbuffpos + 3])
        Cij_r1c4vec = fma(Aip_vec1, Bpj_c4vec, Cij_r1c4vec)
        Cij_r2c4vec = fma(Aip_vec2, Bpj_c4vec, Cij_r2c4vec)
        # Cij[1:8, 5] += A[i:(i+7), p] * B[p, j+4]
        Bpj_c5vec = Vec{4,T}(@inbounds Bbuff[indBbuffpos + 4])
        Cij_r1c5vec = fma(Aip_vec1, Bpj_c5vec, Cij_r1c5vec)
        Cij_r2c5vec = fma(Aip_vec2, Bpj_c5vec, Cij_r2c5vec)
        # Cij[1:8, 6] += A[i:(i+7), p] * B[p, j+5]
        Bpj_c6vec = Vec{4,T}(@inbounds Bbuff[indBbuffpos + 5])
        Cij_r1c6vec = fma(Aip_vec1, Bpj_c6vec, Cij_r1c6vec)
        Cij_r2c6vec = fma(Aip_vec2, Bpj_c6vec, Cij_r2c6vec)
        ptrAbuffpos += microM*sizeof(T)
        indBbuffpos += microN
    end

    # store Cij[:, 1]
    vstore(Cij_r1c1vec, ptrCij              ) # C[i  , j]
    vstore(Cij_r2c1vec, ptrCij + 4*sizeof(T)) # C[i+4, j]
    # store Cij[:, 2]
    vstore(Cij_r1c2vec, ptrCij +       m*sizeof(T)) # C[i  , j+1]
    vstore(Cij_r2c2vec, ptrCij + (4 + m)*sizeof(T)) # C[i+4, j+1]
    # store Cij[:, 3]
    vstore(Cij_r1c3vec, ptrCij +       2m*sizeof(T)) # C[i  , j+2]
    vstore(Cij_r2c3vec, ptrCij + (4 + 2m)*sizeof(T)) # C[i+4, j+2]
    # store Cij[:, 4]
    vstore(Cij_r1c4vec, ptrCij +       3m*sizeof(T)) # C[i  , j+3]
    vstore(Cij_r2c4vec, ptrCij + (4 + 3m)*sizeof(T)) # C[i+4, j+3]
    # store Cij[:, 5]
    vstore(Cij_r1c5vec, ptrCij +       4m*sizeof(T)) # C[i  , j+4]
    vstore(Cij_r2c5vec, ptrCij + (4 + 4m)*sizeof(T)) # C[i+4, j+4]
    # store Cij[:, 6]
    vstore(Cij_r1c6vec, ptrCij +       5m*sizeof(T)) # C[i  , j+5]
    vstore(Cij_r2c6vec, ptrCij + (4 + 5m)*sizeof(T)) # C[i+4, j+5]
    return nothing
end
function packAbuffer!(Abuff, A, blockistart, blockiend, blockpstart, blockpend, microM)
    l = 0 # write position in packing buffer
    for panelistart in blockistart:microM:blockiend # iterate over row panels
        paneliend = panelistart + microM - 1
        if paneliend <= blockiend # row panel is full height
            for p in blockpstart:blockpend # iterate through panel cols
                for i in panelistart:paneliend # iterate though panel rows
                    @inbounds Abuff[l += 1] = A[i, p]
                end
            end
        else # row panel is not full height
            for p in blockpstart:blockpend # iterate through panel cols
                for i in panelistart:blockiend # iterate through live panel rows
                    @inbounds Abuff[l += 1] = A[i, p]
                end
                for i in (blockiend + 1):paneliend
                    @inbounds Abuff[l += 1] = zero(eltype(Abuff))
                end
            end
        end
    end
    return nothing
end
function packBbuffer!(Bbuff, B, blockpstart, blockpend, blockjstart, blockjend, microN)
    l = 0 # write position in packing buffer
    for paneljstart in blockjstart:microN:blockjend # iterate over column panels
        paneljend = paneljstart + microN - 1
        if paneljend <= blockjend # column panel is full width
            for p in blockpstart:blockpend # iterate through panel rows
                for j in paneljstart:paneljend # iterate through panel cols
                    @inbounds Bbuff[l += 1] = B[p, j]
                end
            end
        else # column panel is not full width
            for p in blockpstart:blockpend # iterate through panel rows
                for j in paneljstart:blockjend # iterate through live panel cols
                    @inbounds Bbuff[l += 1] = B[p, j]
                end
                for j in (blockjend + 1):paneljend # zero pad for absent cols
                    @inbounds Bbuff[l += 1] = zero(eltype(Bbuff))
                end
            end
        end
    end
    return nothing
end

# select implementation (packing, non-packing) dependent upon whether the problem roughly fits in L3 caches
gemm_switchpacking!(C, A, B, Abuff, Bbuff, cachePsnp, cachePsp, ::Val{microM}, ::Val{microN}) where {microM,microN} =
    (sizeof(C) + sizeof(A) + sizeof(B)) < 4*2^20 ?
        gemm_nonpacking!(C, A, B, cachePsnp, Val(microM), Val(microN)) :
        gemm_packing!(C, A, B, Abuff, Bbuff, cachePsp, Val(microM), Val(microN))

# check correctness
using Test
let
    m, n, k = 48 .* (3, 2, 1)
    C = rand(m, n)
    A = rand(m, k)
    B = rand(k, n)
    Cref = A * B
    # simple three-nested-loop impelmentations
    @test gemm_jip!(fill!(C, 0), A, B) ≈ Cref
    # non-packing implementation
    microM, microN = 12, 4
    cacheM, cacheN, cacheK = 72, 72, 4080
    cachePsnp = (cacheM, cacheN, cacheK)
    @test gemm_nonpacking!(fill!(C, 0), A, B, cachePsnp, Val(microM), Val(microN)) ≈ Cref
    # packing implementation, 12x4
    microM, microN = 12, 4
    cacheM, cacheN, cacheK = 72, 192, 4080
    cachePsp = (cacheM, cacheN, cacheK)
    Abuff = zeros(cacheM*cacheK)
    Bbuff = zeros(cacheK*cacheN)
    @test gemm_packing!(fill!(C, 0), A, B, Abuff, Bbuff, cachePsp, Val(microM), Val(microN)) ≈ Cref
    # hybrid
    @test gemm_switchpacking!(fill!(C, 0), A, B, Abuff, Bbuff, cachePsnp, cachePsp, Val(microM), Val(microN)) ≈ Cref
    # packing implementation, 8x6
    microM, microN = 8, 6
    cacheM, cacheN, cacheK = 72, 192, 4080
    cachePsp = (cacheM, cacheN, cacheK)
    Abuff = zeros(cacheM*cacheK)
    Bbuff = zeros(cacheK*cacheN)
    @test gemm_packing!(fill!(C, 0), A, B, Abuff, Bbuff, cachePsp, Val(microM), Val(microN)) ≈ Cref
end


# generate benchmark data
using LinearAlgebra
using BenchmarkTools
LinearAlgebra.BLAS.set_num_threads(1) # single-thread OpenBLAS
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5 # constrain bench time

mnks = 48*(1:2:30)
foo = zeros(length(mnks));
# reference implementation (openblas)
timings_ref = copy(foo);
# three-nested-loop implementation
timings_jip = copy(foo);
# non-packing implementation
timings_nonpacking_72x72x4080_12x4 = copy(foo);
# packing implementation
timings_packing_72x192x4080_12x4 = copy(foo);
# timings_packing_72x192x4080_8x6 = copy(foo)
# switch packing implementations
timings_switchpacking_12x4 = copy(foo);
# timings_switchpacking_8x6 = copy(foo)
for (mnkind, mnk) in enumerate(mnks)
    A = rand(mnk, mnk)
    B = rand(mnk, mnk)
    C = rand(mnk, mnk)
    print("Benchmarking with $mnk-by-$mnk matrices..."); @time begin
        # reference implementation (openblas)
        timings_ref[mnkind] = @belapsed mul!($C, $A, $B)
        # three-nested-loop implementation
        timings_jip[mnkind] = @belapsed gemm_jip!($C, $A, $B)
        # non-packing implementation
        microM, microN = 12, 4
        cacheM, cacheN, cacheK = 72, 72, 4080
        cachePsnp = (cacheM, cacheN, cacheK)
        timings_nonpacking_72x72x4080_12x4[mnkind] =
            @belapsed gemm_nonpacking!($C, $A, $B, $cachePsnp, $(Val(microM)), $(Val(microN)))
        # packed implementation, 12x4
        microM, microN = 12, 4
        cacheM, cacheN, cacheK = 72, 192, 4080
        cachePsp = (cacheM, cacheN, cacheK)
        Abuff = zeros(cacheM*cacheK)
        Bbuff = zeros(cacheK*cacheN)
        timings_packing_72x192x4080_12x4[mnkind] =
            @belapsed gemm_packing!($C, $A, $B, $Abuff, $Bbuff, $cachePsp, $(Val(microM)), $(Val(microN)))
        # # packed implementation, 8x6
        # microM, microN = 8, 6
        # cacheM, cacheN, cacheK = 72, 192, 4080
        # cachePsp = (cacheM, cacheN, cacheK)
        # Abuff = zeros(cacheM*cacheK)
        # Bbuff = zeros(cacheK*cacheN)
        # timings_packing_72x192x4080_8x6[mnkind] =
        #     @belapsed gemm_packing!($C, $A, $B, $Abuff, $Bbuff, $cachePsp, $(Val(microM)), $(Val(microN)))
        # packing switch implementation
        microM, microN = 12, 4
        Abuff = zeros(cacheM*cacheK)
        Bbuff = zeros(cacheK*cacheN)
        timings_switchpacking_12x4[mnkind] =
            @belapsed gemm_switchpacking!($C, $A, $B, $Abuff, $Bbuff, $cachePsnp, $cachePsp, $(Val(microM)), $(Val(microN)))
    end
end;


# visualize benchmark data
using Gaston
Gaston.set(linewidth = 4)
Gaston.set(terminal = "qt")
Gaston.gnuplot_send("set term qt font \"sans,14\"")

tableaurgb = ((255, 158,  74), (237, 102,  93),
    (173, 139, 201), (114, 158, 206), (103, 191,  92), (237, 151, 202),
    (205, 204,  93), (168, 120, 110), (162, 162, 162), (109, 204, 218))
tableauhex = (rgb -> string("0x", string.(rgb; base = 16, pad = 2)...)).(tableaurgb)

# visualize timings
ylims = extrema(vcat(timings_ref, timings_jip,
    timings_nonpacking_72x72x4080_12x4,
    timings_packing_72x192x4080_12x4,
    # timings_packing_72x192x4080_8x6,
    timings_switchpacking_12x4,
    ))
plot(mnks, timings_ref, legend = "ref", color = "black",
    xlabel = "matrix dimensions (m = n = k)",
    ylabel = "minimum sample time (seconds)",
    xrange = "[$(first(mnks)):$(last(mnks))]",
    yrange = "[$(first(ylims)):$(last(ylims))]",)
plot!(mnks, timings_jip, legend = "jip", color = "black", linestyle="-")
# non-packing implementations
plot!(mnks, timings_nonpacking_72x72x4080_12x4, legend = "nonpacking\\_72x72x4080\\_12x4", color = tableauhex[1], linestyle=".")
# packing implementations
plot!(mnks, timings_packing_72x192x4080_12x4, legend = "packing\\_72x192x4080\\_12x4", color = tableauhex[3], linestyle=".")
# plot!(mnks, timings_packing_72x192x4080_8x6, legend = "packing\\_72x192x4080\\_8x6", color = tableauhex[3], linestyle=".")
# switch packing implementations
plot!(mnks, timings_switchpacking_12x4, legend = "switchpacking\\_12x4", color = tableauhex[4])

# visualize gflops
hidereference = false
timetogflops(mnk, time) = 2 * mnk^3 / time / 10^9
# reference implementation (openblas)
gflops_ref = timetogflops.(mnks, timings_ref)
# three-nested-loop implementation
gflops_jip = timetogflops.(mnks, timings_jip)
# non-packing implementations
gflops_nonpacking_72x72x4080_12x4 = timetogflops.(mnks, timings_nonpacking_72x72x4080_12x4)
# packing implementations
gflops_packing_72x192x4080_12x4 = timetogflops.(mnks, timings_packing_72x192x4080_12x4)
# gflops_packing_72x192x4080_8x6 = timetogflops.(mnks, timings_packing_72x192x4080_8x6)
# switch packing implementations
gflops_switchpacking_12x4 = timetogflops.(mnks, timings_switchpacking_12x4)
ylims = [0, maximum(vcat(gflops_jip, (hidereference ? [] : gflops_ref),
    gflops_nonpacking_72x72x4080_12x4,
    gflops_packing_72x192x4080_12x4,
    # gflops_packing_72x192x4080_8x6,
    gflops_switchpacking_12x4,
    ))]
plot(mnks, gflops_jip, legend = "jip", color = "black", linestyle=".", linewidth=1,
    xlabel = "matrix dimensions (m = n = k)",
    ylabel = "GFLOPS from minimum sample time",
    xrange = "[$(first(mnks)):$(last(mnks))]",
    yrange = "[$(first(ylims)):$(last(ylims))]",
    box = "bottom right",)
# non-packing implementations
plot!(mnks, gflops_nonpacking_72x72x4080_12x4,
    legend = "nonpacking\\_72x72x4080\\_12x4",
    color = tableauhex[1], linestyle="..", linewidth = 3)
# packing implementations
plot!(mnks, gflops_packing_72x192x4080_12x4,
    legend = "packing\\_72x192x4080\\_12x4",
    color = tableauhex[3], linestyle="..", linewidth = 3)
# plot!(mnks, gflops_packing_72x192x4080_8x6,
#     legend = "packing\\_72x192x4080\\_8x6",
#     color = tableauhex[3], linestyle="..", linewidth = 3)
# switch packing implementations
plot!(mnks, gflops_switchpacking_12x4, legend = "switchpacking\\_12x4", color = tableauhex[4])
# reference
hidereference || plot!(mnks, gflops_ref, legend = "ref", color = "black")
