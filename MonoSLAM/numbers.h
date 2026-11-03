#ifndef MONOSLAM_NUMBERS_H
#define MONOSLAM_NUMBERS_H

namespace monoslam::numbers {

inline constexpr double PI = 3.14159265358979323846;
inline constexpr double PI_DOUBLE = 2.0 * PI;
inline constexpr double RAD_TO_DEG = 180.0 / PI;
inline constexpr double DEG_TO_RAD = PI / 180.0;

inline constexpr float PI_F = static_cast<float>(PI);
inline constexpr float PI_DOUBLE_F = static_cast<float>(PI_DOUBLE);
inline constexpr float RAD_TO_DEG_F = static_cast<float>(RAD_TO_DEG);
inline constexpr float DEG_TO_RAD_F = static_cast<float>(DEG_TO_RAD);

}

#endif
