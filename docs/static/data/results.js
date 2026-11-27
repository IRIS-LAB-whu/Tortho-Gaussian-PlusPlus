window.PROJECT_DATA = {
  "scenes": [
    {
      "name": "Chengzi",
      "dataset": "Custom-collected",
      "images": 997,
      "views": "Oblique imagery",
      "altitude": "Stable flight altitude",
      "src": "static/images/scene-00.webp",
      "width": 2000,
      "height": 881
    },
    {
      "name": "DJI",
      "dataset": "Custom-collected",
      "images": 874,
      "views": "Nadir imagery",
      "altitude": "Stable flight altitude",
      "src": "static/images/scene-01.webp",
      "width": 1934,
      "height": 2000
    },
    {
      "name": "Tianjin_Xiqing",
      "dataset": "Custom-collected",
      "images": 305,
      "views": "Nadir + oblique imagery",
      "altitude": "Stable flight altitude",
      "src": "static/images/scene-02.webp",
      "width": 1823,
      "height": 2000
    },
    {
      "name": "ieu",
      "dataset": "NPU DroneMap",
      "images": 467,
      "views": "Nadir imagery",
      "altitude": "Stable flight altitude",
      "src": "static/images/scene-03.webp",
      "width": 1673,
      "height": 2000
    },
    {
      "name": "npu",
      "dataset": "NPU DroneMap",
      "images": 457,
      "views": "Nadir imagery",
      "altitude": "Stable flight altitude",
      "src": "static/images/scene-04.webp",
      "width": 1864,
      "height": 2000
    },
    {
      "name": "CUHK_LOWER",
      "dataset": "GauU-Scene V2",
      "images": 670,
      "views": "Nadir + oblique imagery",
      "altitude": "Stable flight altitude",
      "src": "static/images/scene-05.webp",
      "width": 1486,
      "height": 797
    },
    {
      "name": "CUHK_UPPER",
      "dataset": "GauU-Scene V2",
      "images": 715,
      "views": "Nadir + oblique imagery",
      "altitude": "Stable flight altitude",
      "src": "static/images/scene-06.webp",
      "width": 2000,
      "height": 1993
    },
    {
      "name": "SMBU",
      "dataset": "GauU-Scene V2",
      "images": 563,
      "views": "Nadir + oblique imagery",
      "altitude": "Stable flight altitude",
      "src": "static/images/scene-07.webp",
      "width": 2000,
      "height": 1813
    },
    {
      "name": "Building",
      "dataset": "Mill 19",
      "images": 1940,
      "views": "Oblique imagery",
      "altitude": "Stable flight altitude",
      "src": "static/images/scene-08.webp",
      "width": 1485,
      "height": 2000
    },
    {
      "name": "Rubble",
      "dataset": "Mill 19",
      "images": 1678,
      "views": "Oblique imagery",
      "altitude": "Stable flight altitude",
      "src": "static/images/scene-09.webp",
      "width": 1781,
      "height": 2000
    },
    {
      "name": "Residence",
      "dataset": "UrbanScene3D",
      "images": 2582,
      "views": "Oblique imagery",
      "altitude": "Variable flight altitude",
      "src": "static/images/scene-10.webp",
      "width": 1872,
      "height": 2000
    },
    {
      "name": "Sci-Art",
      "dataset": "UrbanScene3D",
      "images": 3019,
      "views": "Oblique imagery",
      "altitude": "Variable flight altitude",
      "src": "static/images/scene-11.webp",
      "width": 1874,
      "height": 2000
    }
  ],
  "figures": {
    "teaser": {
      "src": "static/images/teaser.webp",
      "width": 2717,
      "height": 2704
    },
    "framework": {
      "src": "static/images/framework.webp",
      "width": 3200,
      "height": 2297
    },
    "drift": {
      "src": "static/images/drift.webp",
      "width": 2297,
      "height": 3200
    },
    "facades": {
      "src": "static/images/facades.webp",
      "width": 3184,
      "height": 3200
    },
    "gravity": {
      "src": "static/images/gravity.webp",
      "width": 3200,
      "height": 3171
    },
    "gravity-fusion": {
      "src": "static/images/gravity-fusion.webp",
      "width": 3200,
      "height": 1519
    },
    "adam": {
      "src": "static/images/adam.webp",
      "width": 2460,
      "height": 1110
    },
    "threshold": {
      "src": "static/images/threshold.webp",
      "width": 3200,
      "height": 1712
    }
  },
  "rows": [
    {
      "scene": "Chengzi",
      "method": "3DGS",
      "status": "ok",
      "psnr": 25.04,
      "ssim": 0.874,
      "lpips": 0.157,
      "time": 53.0,
      "gaussians": 14.63,
      "vram": 23.03
    },
    {
      "scene": "DJI",
      "method": "3DGS",
      "status": "ok",
      "psnr": 24.54,
      "ssim": 0.863,
      "lpips": 0.104,
      "time": 62.38,
      "gaussians": 17.89,
      "vram": 29.07
    },
    {
      "scene": "Tianjin_Xiqing",
      "method": "3DGS",
      "status": "ok",
      "psnr": 26.56,
      "ssim": 0.866,
      "lpips": 0.124,
      "time": 30.08,
      "gaussians": 4.56,
      "vram": 7.29
    },
    {
      "scene": "Chengzi",
      "method": "CityGaussian",
      "status": "ok",
      "psnr": 26.01,
      "ssim": 0.893,
      "lpips": 0.119,
      "time": 342.93,
      "gaussians": 19.91,
      "vram": 28.19
    },
    {
      "scene": "DJI",
      "method": "CityGaussian",
      "status": "OOM",
      "psnr": null,
      "ssim": null,
      "lpips": null,
      "time": null,
      "gaussians": null,
      "vram": null
    },
    {
      "scene": "Tianjin_Xiqing",
      "method": "CityGaussian",
      "status": "ok",
      "psnr": 26.52,
      "ssim": 0.865,
      "lpips": 0.118,
      "time": 309.48,
      "gaussians": 5.01,
      "vram": 8.87
    },
    {
      "scene": "Chengzi",
      "method": "FastGS",
      "status": "ok",
      "psnr": 28.39,
      "ssim": 0.888,
      "lpips": 0.184,
      "time": 4.28,
      "gaussians": 1.15,
      "vram": 2.15
    },
    {
      "scene": "DJI",
      "method": "FastGS",
      "status": "ok",
      "psnr": 26.84,
      "ssim": 0.891,
      "lpips": 0.117,
      "time": 3.23,
      "gaussians": 0.56,
      "vram": 0.99
    },
    {
      "scene": "Tianjin_Xiqing",
      "method": "FastGS",
      "status": "ok",
      "psnr": 26.45,
      "ssim": 0.852,
      "lpips": 0.173,
      "time": 6.75,
      "gaussians": 0.53,
      "vram": 1.1
    },
    {
      "scene": "Chengzi",
      "method": "Tortho-Gaussian",
      "status": "ok",
      "psnr": 25.2,
      "ssim": 0.875,
      "lpips": 0.156,
      "time": 79.35,
      "gaussians": 19.2,
      "vram": 17.03
    },
    {
      "scene": "DJI",
      "method": "Tortho-Gaussian",
      "status": "OOM",
      "psnr": null,
      "ssim": null,
      "lpips": null,
      "time": null,
      "gaussians": null,
      "vram": null
    },
    {
      "scene": "Tianjin_Xiqing",
      "method": "Tortho-Gaussian",
      "status": "ok",
      "psnr": 24.58,
      "ssim": 0.823,
      "lpips": 0.199,
      "time": 40.6,
      "gaussians": 1.42,
      "vram": 3.94
    },
    {
      "scene": "Chengzi",
      "method": "Ortho-3DGS",
      "status": "ok",
      "psnr": 27.85,
      "ssim": 0.895,
      "lpips": 0.143,
      "time": 71.63,
      "gaussians": 16.72,
      "vram": 32.22
    },
    {
      "scene": "DJI",
      "method": "Ortho-3DGS",
      "status": "OOM",
      "psnr": null,
      "ssim": null,
      "lpips": null,
      "time": null,
      "gaussians": null,
      "vram": null
    },
    {
      "scene": "Tianjin_Xiqing",
      "method": "Ortho-3DGS",
      "status": "ok",
      "psnr": 26.58,
      "ssim": 0.866,
      "lpips": 0.124,
      "time": 35.02,
      "gaussians": 4.61,
      "vram": 7.96
    },
    {
      "scene": "Chengzi",
      "method": "Tortho-Gaussian++",
      "status": "ok",
      "psnr": 29.01,
      "ssim": 0.901,
      "lpips": 0.158,
      "time": 33.8,
      "gaussians": 7.15,
      "vram": 11.4
    },
    {
      "scene": "DJI",
      "method": "Tortho-Gaussian++",
      "status": "ok",
      "psnr": 27.49,
      "ssim": 0.914,
      "lpips": 0.083,
      "time": 27.67,
      "gaussians": 6.87,
      "vram": 11.12
    },
    {
      "scene": "Tianjin_Xiqing",
      "method": "Tortho-Gaussian++",
      "status": "ok",
      "psnr": 26.74,
      "ssim": 0.86,
      "lpips": 0.16,
      "time": 18.0,
      "gaussians": 1.15,
      "vram": 3.31
    },
    {
      "scene": "ieu",
      "method": "3DGS",
      "status": "ok",
      "psnr": 32.41,
      "ssim": 0.936,
      "lpips": 0.088,
      "time": 47.05,
      "gaussians": 9.83,
      "vram": 15.42
    },
    {
      "scene": "npu",
      "method": "3DGS",
      "status": "ok",
      "psnr": 36.38,
      "ssim": 0.968,
      "lpips": 0.04,
      "time": 40.72,
      "gaussians": 7.46,
      "vram": 12.68
    },
    {
      "scene": "CUHK_LOWER",
      "method": "3DGS",
      "status": "ok",
      "psnr": 27.56,
      "ssim": 0.892,
      "lpips": 0.128,
      "time": 32.97,
      "gaussians": 6.96,
      "vram": 10.64
    },
    {
      "scene": "ieu",
      "method": "CityGaussian",
      "status": "Failed",
      "psnr": null,
      "ssim": null,
      "lpips": null,
      "time": null,
      "gaussians": null,
      "vram": null
    },
    {
      "scene": "npu",
      "method": "CityGaussian",
      "status": "Failed",
      "psnr": null,
      "ssim": null,
      "lpips": null,
      "time": null,
      "gaussians": null,
      "vram": null
    },
    {
      "scene": "CUHK_LOWER",
      "method": "CityGaussian",
      "status": "ok",
      "psnr": 28.08,
      "ssim": 0.9,
      "lpips": 0.107,
      "time": 207.92,
      "gaussians": 8.32,
      "vram": 12.72
    },
    {
      "scene": "ieu",
      "method": "FastGS",
      "status": "ok",
      "psnr": 32.49,
      "ssim": 0.921,
      "lpips": 0.133,
      "time": 4.34,
      "gaussians": 0.67,
      "vram": 1.25
    },
    {
      "scene": "npu",
      "method": "FastGS",
      "status": "ok",
      "psnr": 35.27,
      "ssim": 0.956,
      "lpips": 0.07,
      "time": 4.24,
      "gaussians": 0.41,
      "vram": 0.84
    },
    {
      "scene": "CUHK_LOWER",
      "method": "FastGS",
      "status": "ok",
      "psnr": 27.74,
      "ssim": 0.871,
      "lpips": 0.184,
      "time": 4.68,
      "gaussians": 0.76,
      "vram": 1.51
    },
    {
      "scene": "ieu",
      "method": "Tortho-Gaussian",
      "status": "ok",
      "psnr": 32.69,
      "ssim": 0.936,
      "lpips": 0.088,
      "time": 48.94,
      "gaussians": 4.84,
      "vram": 8.11
    },
    {
      "scene": "npu",
      "method": "Tortho-Gaussian",
      "status": "ok",
      "psnr": 36.27,
      "ssim": 0.968,
      "lpips": 0.04,
      "time": 43.18,
      "gaussians": 4.99,
      "vram": 8.74
    },
    {
      "scene": "CUHK_LOWER",
      "method": "Tortho-Gaussian",
      "status": "ok",
      "psnr": 27.47,
      "ssim": 0.891,
      "lpips": 0.129,
      "time": 78.8,
      "gaussians": 11.4,
      "vram": 27.66
    },
    {
      "scene": "ieu",
      "method": "Ortho-3DGS",
      "status": "ok",
      "psnr": 32.74,
      "ssim": 0.936,
      "lpips": 0.086,
      "time": 54.59,
      "gaussians": 11.07,
      "vram": 18.83
    },
    {
      "scene": "npu",
      "method": "Ortho-3DGS",
      "status": "ok",
      "psnr": 36.44,
      "ssim": 0.967,
      "lpips": 0.04,
      "time": 42.91,
      "gaussians": 8.84,
      "vram": 15.35
    },
    {
      "scene": "CUHK_LOWER",
      "method": "Ortho-3DGS",
      "status": "ok",
      "psnr": 28.24,
      "ssim": 0.896,
      "lpips": 0.12,
      "time": 49.96,
      "gaussians": 8.69,
      "vram": 13.39
    },
    {
      "scene": "ieu",
      "method": "Tortho-Gaussian++",
      "status": "ok",
      "psnr": 32.79,
      "ssim": 0.932,
      "lpips": 0.104,
      "time": 20.28,
      "gaussians": 3.22,
      "vram": 6.3
    },
    {
      "scene": "npu",
      "method": "Tortho-Gaussian++",
      "status": "ok",
      "psnr": 36.5,
      "ssim": 0.969,
      "lpips": 0.045,
      "time": 21.62,
      "gaussians": 2.82,
      "vram": 5.88
    },
    {
      "scene": "CUHK_LOWER",
      "method": "Tortho-Gaussian++",
      "status": "ok",
      "psnr": 28.4,
      "ssim": 0.899,
      "lpips": 0.14,
      "time": 16.25,
      "gaussians": 2.63,
      "vram": 6.07
    },
    {
      "scene": "CUHK_UPPER",
      "method": "3DGS",
      "status": "ok",
      "psnr": 26.82,
      "ssim": 0.886,
      "lpips": 0.114,
      "time": 49.78,
      "gaussians": 7.37,
      "vram": 11.07
    },
    {
      "scene": "SMBU",
      "method": "3DGS",
      "status": "ok",
      "psnr": 25.55,
      "ssim": 0.893,
      "lpips": 0.108,
      "time": 32.08,
      "gaussians": 5.48,
      "vram": 8.56
    },
    {
      "scene": "Building",
      "method": "3DGS",
      "status": "ok",
      "psnr": 22.84,
      "ssim": 0.821,
      "lpips": 0.17,
      "time": 44.97,
      "gaussians": 7.92,
      "vram": 11.88
    },
    {
      "scene": "CUHK_UPPER",
      "method": "CityGaussian",
      "status": "ok",
      "psnr": 27.2,
      "ssim": 0.896,
      "lpips": 0.098,
      "time": 214.05,
      "gaussians": 9.33,
      "vram": 14.41
    },
    {
      "scene": "SMBU",
      "method": "CityGaussian",
      "status": "ok",
      "psnr": 25.44,
      "ssim": 0.895,
      "lpips": 0.1,
      "time": 173.1,
      "gaussians": 5.95,
      "vram": 9.51
    },
    {
      "scene": "Building",
      "method": "CityGaussian",
      "status": "ok",
      "psnr": 23.44,
      "ssim": 0.846,
      "lpips": 0.145,
      "time": 357.02,
      "gaussians": 9.24,
      "vram": 14.79
    },
    {
      "scene": "CUHK_UPPER",
      "method": "FastGS",
      "status": "ok",
      "psnr": 26.75,
      "ssim": 0.858,
      "lpips": 0.179,
      "time": 4.95,
      "gaussians": 0.85,
      "vram": 1.46
    },
    {
      "scene": "SMBU",
      "method": "FastGS",
      "status": "ok",
      "psnr": 25.26,
      "ssim": 0.867,
      "lpips": 0.163,
      "time": 5.03,
      "gaussians": 0.67,
      "vram": 1.27
    },
    {
      "scene": "Building",
      "method": "FastGS",
      "status": "ok",
      "psnr": 22.88,
      "ssim": 0.798,
      "lpips": 0.223,
      "time": 21.37,
      "gaussians": 0.74,
      "vram": 2.55
    },
    {
      "scene": "CUHK_UPPER",
      "method": "Tortho-Gaussian",
      "status": "ok",
      "psnr": 26.79,
      "ssim": 0.885,
      "lpips": 0.115,
      "time": 47.51,
      "gaussians": 8.68,
      "vram": 14.61
    },
    {
      "scene": "SMBU",
      "method": "Tortho-Gaussian",
      "status": "ok",
      "psnr": 25.49,
      "ssim": 0.883,
      "lpips": 0.107,
      "time": 46.52,
      "gaussians": 4.97,
      "vram": 10.22
    },
    {
      "scene": "Building",
      "method": "Tortho-Gaussian",
      "status": "ok",
      "psnr": 22.77,
      "ssim": 0.82,
      "lpips": 0.178,
      "time": 86.94,
      "gaussians": 11.06,
      "vram": 17.69
    },
    {
      "scene": "CUHK_UPPER",
      "method": "Ortho-3DGS",
      "status": "ok",
      "psnr": 27.91,
      "ssim": 0.891,
      "lpips": 0.11,
      "time": 44.78,
      "gaussians": 8.49,
      "vram": 14.57
    },
    {
      "scene": "SMBU",
      "method": "Ortho-3DGS",
      "status": "ok",
      "psnr": 26.76,
      "ssim": 0.898,
      "lpips": 0.104,
      "time": 35.62,
      "gaussians": 5.95,
      "vram": 9.24
    },
    {
      "scene": "Building",
      "method": "Ortho-3DGS",
      "status": "ok",
      "psnr": 23.35,
      "ssim": 0.81,
      "lpips": 0.179,
      "time": 71.43,
      "gaussians": 9.78,
      "vram": 14.61
    },
    {
      "scene": "CUHK_UPPER",
      "method": "Tortho-Gaussian++",
      "status": "ok",
      "psnr": 27.6,
      "ssim": 0.886,
      "lpips": 0.136,
      "time": 15.35,
      "gaussians": 2.35,
      "vram": 5.36
    },
    {
      "scene": "SMBU",
      "method": "Tortho-Gaussian++",
      "status": "ok",
      "psnr": 26.24,
      "ssim": 0.89,
      "lpips": 0.127,
      "time": 14.65,
      "gaussians": 1.84,
      "vram": 4.58
    },
    {
      "scene": "Building",
      "method": "Tortho-Gaussian++",
      "status": "ok",
      "psnr": 23.9,
      "ssim": 0.826,
      "lpips": 0.17,
      "time": 20.64,
      "gaussians": 3.7,
      "vram": 7.65
    },
    {
      "scene": "Rubble",
      "method": "3DGS",
      "status": "ok",
      "psnr": 26.87,
      "ssim": 0.87,
      "lpips": 0.161,
      "time": 28.38,
      "gaussians": 5.86,
      "vram": 9.0
    },
    {
      "scene": "Residence",
      "method": "3DGS",
      "status": "ok",
      "psnr": 22.43,
      "ssim": 0.852,
      "lpips": 0.152,
      "time": 29.65,
      "gaussians": 5.69,
      "vram": 8.98
    },
    {
      "scene": "Sci-Art",
      "method": "3DGS",
      "status": "ok",
      "psnr": 22.16,
      "ssim": 0.867,
      "lpips": 0.144,
      "time": 19.87,
      "gaussians": 3.07,
      "vram": 5.24
    },
    {
      "scene": "Rubble",
      "method": "CityGaussian",
      "status": "ok",
      "psnr": 27.52,
      "ssim": 0.888,
      "lpips": 0.137,
      "time": 308.38,
      "gaussians": 6.35,
      "vram": 11.98
    },
    {
      "scene": "Residence",
      "method": "CityGaussian",
      "status": "ok",
      "psnr": 22.69,
      "ssim": 0.861,
      "lpips": 0.142,
      "time": 362.55,
      "gaussians": 6.01,
      "vram": 10.94
    },
    {
      "scene": "Sci-Art",
      "method": "CityGaussian",
      "status": "ok",
      "psnr": 22.36,
      "ssim": 0.875,
      "lpips": 0.135,
      "time": 255.63,
      "gaussians": 2.95,
      "vram": 5.41
    },
    {
      "scene": "Rubble",
      "method": "FastGS",
      "status": "ok",
      "psnr": 25.33,
      "ssim": 0.812,
      "lpips": 0.247,
      "time": 20.22,
      "gaussians": 0.75,
      "vram": 2.62
    },
    {
      "scene": "Residence",
      "method": "FastGS",
      "status": "ok",
      "psnr": 22.06,
      "ssim": 0.825,
      "lpips": 0.195,
      "time": 34.62,
      "gaussians": 0.77,
      "vram": 3.05
    },
    {
      "scene": "Sci-Art",
      "method": "FastGS",
      "status": "ok",
      "psnr": 21.55,
      "ssim": 0.837,
      "lpips": 0.191,
      "time": 41.1,
      "gaussians": 0.57,
      "vram": 3.54
    },
    {
      "scene": "Rubble",
      "method": "Tortho-Gaussian",
      "status": "ok",
      "psnr": 26.65,
      "ssim": 0.849,
      "lpips": 0.196,
      "time": 59.95,
      "gaussians": 6.59,
      "vram": 9.48
    },
    {
      "scene": "Residence",
      "method": "Tortho-Gaussian",
      "status": "ok",
      "psnr": 22.49,
      "ssim": 0.852,
      "lpips": 0.152,
      "time": 83.16,
      "gaussians": 6.56,
      "vram": 10.22
    },
    {
      "scene": "Sci-Art",
      "method": "Tortho-Gaussian",
      "status": "ok",
      "psnr": 22.18,
      "ssim": 0.867,
      "lpips": 0.145,
      "time": 85.04,
      "gaussians": 2.46,
      "vram": 4.57
    },
    {
      "scene": "Rubble",
      "method": "Ortho-3DGS",
      "status": "ok",
      "psnr": 27.1,
      "ssim": 0.872,
      "lpips": 0.16,
      "time": 54.43,
      "gaussians": 5.92,
      "vram": 9.13
    },
    {
      "scene": "Residence",
      "method": "Ortho-3DGS",
      "status": "ok",
      "psnr": 22.57,
      "ssim": 0.853,
      "lpips": 0.151,
      "time": 74.22,
      "gaussians": 5.75,
      "vram": 9.11
    },
    {
      "scene": "Sci-Art",
      "method": "Ortho-3DGS",
      "status": "ok",
      "psnr": 22.39,
      "ssim": 0.869,
      "lpips": 0.143,
      "time": 63.61,
      "gaussians": 3.13,
      "vram": 5.34
    },
    {
      "scene": "Rubble",
      "method": "Tortho-Gaussian++",
      "status": "ok",
      "psnr": 27.05,
      "ssim": 0.87,
      "lpips": 0.161,
      "time": 29.7,
      "gaussians": 2.97,
      "vram": 6.54
    },
    {
      "scene": "Residence",
      "method": "Tortho-Gaussian++",
      "status": "ok",
      "psnr": 22.59,
      "ssim": 0.844,
      "lpips": 0.167,
      "time": 20.88,
      "gaussians": 3.06,
      "vram": 6.7
    },
    {
      "scene": "Sci-Art",
      "method": "Tortho-Gaussian++",
      "status": "ok",
      "psnr": 22.21,
      "ssim": 0.859,
      "lpips": 0.158,
      "time": 17.97,
      "gaussians": 2.28,
      "vram": 5.78
    }
  ]
};
