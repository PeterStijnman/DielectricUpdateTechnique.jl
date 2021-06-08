"""
copyVec!(dst, src).\n
copies the vector components from source to the destination.\n
dts.x = src.x.\n
dts.y = src.y.\n
dts.z = src.z.\n
"""
function copyVec!(dst, src)
    @inbounds for i in keys(src)
        copyto!(dst[i], src[i])
    end
end

"""
l = lengthVec(x).\n
returns the length (l) the vector components summed.\n
"""
function lengthVec(x)
    l = 0
    @inbounds for i in keys(x)
        l += length(x[i])
    end
    return l
end

"""
axpyVec!(a, x, y).\n
calculates y = a*x + y, for all the vector components.\n
"""
function axpyVec!(a, x, y)
    @inbounds for i in keys(x)
        axpy!(a, x[i], y[i])
    end
end

"""
axpbyVec!(a, x, b, y).\n
calculates y = a*x + b*y, for all the vector components.\n
"""
function axpbyVec!(a, x, b, y)
    @inbounds for i in keys(x)
        axpby!(a, x[i], b, y[i])
    end
end

"""
scaleVec!(a, x).\n
calculates x = a*x, for all the vector components.\n
"""
function scaleVec!(a, x)
    @inbounds for i in keys(x)
        @. x[i] *= a
    end
end

"""
caxpyVec!(c, a, x, y).\n
calculates c = a*x + y, for all the vector components.\n
"""
function caxpyVec!(c, a, x, y)
    copyVec!(c, y)
    axpyVec!(a, x, c)
end