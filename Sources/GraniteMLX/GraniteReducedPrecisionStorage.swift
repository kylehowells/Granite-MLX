import MLX

/// Experimental evaluated storage for activations between encoder layers.
/// Compute still occurs in FP16 on hardware without native FP8/INT8 QQMM.
enum GraniteReducedPrecisionStorage {
    private static let fp8Header = """
    struct granite_fp8_e4m3 {
      template <typename T>
      granite_fp8_e4m3(T f) {
        uint32_t fp8_max = 543 << 21;
        uint32_t denorm_mask = 141 << 23;
        uint32_t f_bits = as_type<uint32_t>(static_cast<float>(f));
        uint32_t sign = f_bits & 0x80000000;
        f_bits ^= sign;
        if (f_bits >= fp8_max) {
          bits = 0x7E;
        } else if (f_bits < (121 << 23)) {
          f_bits = as_type<uint32_t>(
              as_type<float>(f_bits) + as_type<float>(denorm_mask));
          bits = static_cast<uint8_t>(f_bits - denorm_mask);
        } else {
          uint8_t mant_odd = (f_bits >> 20) & 1;
          f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
          f_bits += mant_odd;
          bits = static_cast<uint8_t>(f_bits >> 20);
        }
        bits |= static_cast<uint8_t>(sign >> 24);
      }

      operator half() {
        uint16_t value = (bits & 127) << 7;
        half converted = as_type<half>(value) * 256.0h;
        return (bits & 128) ? -converted : converted;
      }

      uint8_t bits;
    };
    """

    private static let encodeFP8 = MLXFast.metalKernel(
        name: "granite_encode_fp8_e4m3",
        inputNames: ["input"], outputNames: ["output"],
        source: """
        uint index = thread_position_in_grid.x;
        if (index < count) {
          output[index] = granite_fp8_e4m3(float(input[index])).bits;
        }
        """,
        header: fp8Header
    )

    private static let decodeFP8 = MLXFast.metalKernel(
        name: "granite_decode_fp8_e4m3",
        inputNames: ["input"], outputNames: ["output"],
        source: """
        uint index = thread_position_in_grid.x;
        if (index < count) {
          granite_fp8_e4m3 value(0.0f);
          value.bits = input[index];
          output[index] = half(value);
        }
        """,
        header: fp8Header
    )

    static func roundTrip(_ value: MLXArray, mode: GraniteActivationPrecision) -> MLXArray {
        switch mode {
        case .baseline, .fp16:
            MLX.eval(value)
            return value
        case .fp8Emulated:
            return fp8RoundTrip(value)
        case .int8Emulated:
            return int8RoundTrip(value)
        }
    }

    private static func fp8RoundTrip(_ value: MLXArray) -> MLXArray {
        let half = value.asType(.float16)
        let maximum = MLX.max(abs(half).asType(.float32))
        let scale = MLX.maximum(maximum / 448, MLXArray(1e-8))
        let normalized = half / scale.asType(.float16)
        let count = value.size
        let bytes = encodeFP8(
            [normalized], template: [("count", count)],
            grid: (count, 1, 1), threadGroup: (min(256, count), 1, 1),
            outputShapes: [[count]], outputDTypes: [.uint8]
        )[0]
        MLX.eval(bytes, scale)
        let decoded = decodeFP8(
            [bytes], template: [("count", count)],
            grid: (count, 1, 1), threadGroup: (min(256, count), 1, 1),
            outputShapes: [[count]], outputDTypes: [.float16]
        )[0]
        return (decoded * scale.asType(.float16)).reshaped(value.shape)
    }

    private static func int8RoundTrip(_ value: MLXArray) -> MLXArray {
        let half = value.asType(.float16)
        let maximum = MLX.max(abs(half).asType(.float32))
        let scale = MLX.maximum(maximum / 127, MLXArray(1e-8))
        let bytes = MLX.clip(
            round(half / scale.asType(.float16)), min: -127, max: 127
        ).asType(.int8)
        MLX.eval(bytes, scale)
        return bytes.asType(.float16) * scale.asType(.float16)
    }
}
