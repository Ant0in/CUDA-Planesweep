#pragma once
#include <vector>

/**
 * @brief Inverts a 3x3 matrix stored as a flat vector.
 * @param A Matrix values in row-major order, because tiny matrices do not need a whole library moment.
 * @return The inverse matrix, also row-major and also tiny.
 */
template <typename T>
const inline std::vector<T> inverseMatrix3x3(std::vector<T> &A)
{
    double determinant = 0.0f;

    determinant = (A[0] * A[4] * A[8] + A[3] * A[7] * A[2] + A[1] * A[5] * A[6]) -
                  (A[2] * A[4] * A[6] + A[1] * A[3] * A[8] + A[0] * A[5] * A[6]);

    std::vector<T> inv(9);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            inv[i * 3 + j] =
                ((A[((j + 1) % 3) * 3 + ((i + 1) % 3)] * A[((j + 2) % 3) * 3 + ((i + 2) % 3)]) -
                 (A[((j + 1) % 3) * 3 + ((i + 2) % 3)] * A[((j + 2) % 3) * 3 + ((i + 1) % 3)])) /
                determinant;

    return inv;
}

template <typename T>
struct params
{
    std::vector<T> K;
    std::vector<T> R;
    std::vector<T> t;
    std::vector<T> K_inv;
    std::vector<T> R_inv;
    std::vector<T> t_inv;

    /**
     * @brief Empty params object, mostly here so vectors can create cameras before filling them.
     */
    params(){};

    /**
     * @brief Stores camera matrices and precomputes the inverse bits we use a lot.
     * @param _K Intrinsic matrix.
     * @param _R Rotation matrix.
     * @param _t Translation vector.
     */
    params(std::vector<T> _K, std::vector<T> _R, std::vector<T> _t) : K(_K), R(_R), t(_t)
    {
        K_inv = inverseMatrix3x3<double>(K);
        R_inv = inverseMatrix3x3<double>(R);
        t_inv = {-t[0], -t[1], -t[2]};
    };
};

/**
 * @brief Returns the hardcoded camera calibration setup for the sample images.
 * @return One params object per camera, in the same order as v0, v1, v2, v3.
 */
std::vector<params<double>> get_cam_params();
