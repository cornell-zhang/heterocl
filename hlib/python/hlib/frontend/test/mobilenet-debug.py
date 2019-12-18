from test_wrapper import *
import keras
#from keras.datasets import imagenet
from keras.models import load_model, Model
from frontend import relay_parser
import heterocl as hcl
import hlib
import sys
import scipy
import numpy as np
import numpy.testing as tst
"""
def @main(%input_1: Tensor[(1, 3, 224, 224), float32], %v_param_1: Tensor[(32, 3, 3, 3), float32], %v_param_2: Tensor[(32), float32], %v_param_3: Tensor[(32), float32], %v_param_4: Tensor[(32), float32], %v_param_5: Tensor[(32), float32], %v_param_6: Tensor[(32, 1, 3, 3), float32], %v_param_7: Tensor[(32), float32], %v_param_8: Tensor[(32), float32], %v_param_9: Tensor[(32), float32], %v_param_10: Tensor[(32), float32], %v_param_11: Tensor[(64, 32, 1, 1), float32], %v_param_12: Tensor[(64), float32], %v_param_13: Tensor[(64), float32], %v_param_14: Tensor[(64), float32], %v_param_15: Tensor[(64), float32], %v_param_16: Tensor[(64, 1, 3, 3), float32], %v_param_17: Tensor[(64), float32], %v_param_18: Tensor[(64), float32], %v_param_19: Tensor[(64), float32], %v_param_20: Tensor[(64), float32], %v_param_21: Tensor[(128, 64, 1, 1), float32], %v_param_22: Tensor[(128), float32], %v_param_23: Tensor[(128), float32], %v_param_24: Tensor[(128), float32], %v_param_25: Tensor[(128), float32], %v_param_26: Tensor[(128, 1, 3, 3), float32], %v_param_27: Tensor[(128), float32], %v_param_28: Tensor[(128), float32], %v_param_29: Tensor[(128), float32], %v_param_30: Tensor[(128), float32], %v_param_31: Tensor[(128, 128, 1, 1), float32], %v_param_32: Tensor[(128), float32], %v_param_33: Tensor[(128), float32], %v_param_34: Tensor[(128), float32], %v_param_35: Tensor[(128), float32], %v_param_36: Tensor[(128, 1, 3, 3), float32], %v_param_37: Tensor[(128), float32], %v_param_38: Tensor[(128), float32], %v_param_39: Tensor[(128), float32], %v_param_40: Tensor[(128), float32], %v_param_41: Tensor[(256, 128, 1, 1), float32], %v_param_42: Tensor[(256), float32], %v_param_43: Tensor[(256), float32], %v_param_44: Tensor[(256), float32], %v_param_45: Tensor[(256), float32], %v_param_46: Tensor[(256, 1, 3, 3), float32], %v_param_47: Tensor[(256), float32], %v_param_48: Tensor[(256), float32], %v_param_49: Tensor[(256), float32], %v_param_50: Tensor[(256), float32], %v_param_51: Tensor[(256, 256, 1, 1), float32], %v_param_52: Tensor[(256), float32], %v_param_53: Tensor[(256), float32], %v_param_54: Tensor[(256), float32], %v_param_55: Tensor[(256), float32], %v_param_56: Tensor[(256, 1, 3, 3), float32], %v_param_57: Tensor[(256), float32], %v_param_58: Tensor[(256), float32], %v_param_59: Tensor[(256), float32], %v_param_60: Tensor[(256), float32], %v_param_61: Tensor[(512, 256, 1, 1), float32], %v_param_62: Tensor[(512), float32], %v_param_63: Tensor[(512), float32], %v_param_64: Tensor[(512), float32], %v_param_65: Tensor[(512), float32], %v_param_66: Tensor[(512, 1, 3, 3), float32], %v_param_67: Tensor[(512), float32], %v_param_68: Tensor[(512), float32], %v_param_69: Tensor[(512), float32], %v_param_70: Tensor[(512), float32], %v_param_71: Tensor[(512, 512, 1, 1), float32], %v_param_72: Tensor[(512), float32], %v_param_73: Tensor[(512), float32], %v_param_74: Tensor[(512), float32], %v_param_75: Tensor[(512), float32], %v_param_76: Tensor[(512, 1, 3, 3), float32], %v_param_77: Tensor[(512), float32], %v_param_78: Tensor[(512), float32], %v_param_79: Tensor[(512), float32], %v_param_80: Tensor[(512), float32], %v_param_81: Tensor[(512, 512, 1, 1), float32], %v_param_82: Tensor[(512), float32], %v_param_83: Tensor[(512), float32], %v_param_84: Tensor[(512), float32], %v_param_85: Tensor[(512), float32], %v_param_86: Tensor[(512, 1, 3, 3), float32], %v_param_87: Tensor[(512), float32], %v_param_88: Tensor[(512), float32], %v_param_89: Tensor[(512), float32], %v_param_90: Tensor[(512), float32], %v_param_91: Tensor[(512, 512, 1, 1), float32], %v_param_92: Tensor[(512), float32], %v_param_93: Tensor[(512), float32], %v_param_94: Tensor[(512), float32], %v_param_95: Tensor[(512), float32], %v_param_96: Tensor[(512, 1, 3, 3), float32], %v_param_97: Tensor[(512), float32], %v_param_98: Tensor[(512), float32], %v_param_99: Tensor[(512), float32], %v_param_100: Tensor[(512), float32], %v_param_101: Tensor[(512, 512, 1, 1), float32], %v_param_102: Tensor[(512), float32], %v_param_103: Tensor[(512), float32], %v_param_104: Tensor[(512), float32], %v_param_105: Tensor[(512), float32], %v_param_106: Tensor[(512, 1, 3, 3), float32], %v_param_107: Tensor[(512), float32], %v_param_108: Tensor[(512), float32], %v_param_109: Tensor[(512), float32], %v_param_110: Tensor[(512), float32], %v_param_111: Tensor[(512, 512, 1, 1), float32], %v_param_112: Tensor[(512), float32], %v_param_113: Tensor[(512), float32], %v_param_114: Tensor[(512), float32], %v_param_115: Tensor[(512), float32], %v_param_116: Tensor[(512, 1, 3, 3), float32], %v_param_117: Tensor[(512), float32], %v_param_118: Tensor[(512), float32], %v_param_119: Tensor[(512), float32], %v_param_120: Tensor[(512), float32], %v_param_121: Tensor[(1024, 512, 1, 1), float32], %v_param_122: Tensor[(1024), float32], %v_param_123: Tensor[(1024), float32], %v_param_124: Tensor[(1024), float32], %v_param_125: Tensor[(1024), float32], %v_param_126: Tensor[(1024, 1, 3, 3), float32], %v_param_127: Tensor[(1024), float32], %v_param_128: Tensor[(1024), float32], %v_param_129: Tensor[(1024), float32], %v_param_130: Tensor[(1024), float32], %v_param_131: Tensor[(1024, 1024, 1, 1), float32], %v_param_132: Tensor[(1024), float32], %v_param_133: Tensor[(1024), float32], %v_param_134: Tensor[(1024), float32], %v_param_135: Tensor[(1024), float32], %v_param_136: Tensor[(1000, 1024, 1, 1), float32], %v_param_137: Tensor[(1000), float32]) -> Tensor[(1, 1000), float32] {
  %0 = nn.pad(%input_1, pad_width=[[0, 0], [0, 0], [0, 1], [0, 1]]) /* ty=Tensor[(1, 3, 225, 225), float32] */;
  %1 = nn.conv2d(%0, %v_param_1, strides=[2, 2], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %2 = nn.batch_norm(%1, %v_param_2, %v_param_3, %v_param_4, %v_param_5, epsilon=0.001f) /* ty=(Tensor[(1, 32, 112, 112), float32], Tensor[(32), float32], Tensor[(32), float32]) */;
  %3 = %2.0;
  %4 = clip(%3, a_min=0f, a_max=6f) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %5 = nn.conv2d(%4, %v_param_6, padding=[1, 1], groups=32, channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %6 = nn.batch_norm(%5, %v_param_7, %v_param_8, %v_param_9, %v_param_10, epsilon=0.001f) /* ty=(Tensor[(1, 32, 112, 112), float32], Tensor[(32), float32], Tensor[(32), float32]) */;
  %7 = %6.0;
  %8 = clip(%7, a_min=0f, a_max=6f) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %9 = nn.conv2d(%8, %v_param_11, channels=64, kernel_size=[1, 1]) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %10 = nn.batch_norm(%9, %v_param_12, %v_param_13, %v_param_14, %v_param_15, epsilon=0.001f) /* ty=(Tensor[(1, 64, 112, 112), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
  %11 = %10.0;
  %12 = clip(%11, a_min=0f, a_max=6f) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %13 = nn.pad(%12, pad_width=[[0, 0], [0, 0], [0, 1], [0, 1]]) /* ty=Tensor[(1, 64, 113, 113), float32] */;
  %14 = nn.conv2d(%13, %v_param_16, strides=[2, 2], groups=64, channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %15 = nn.batch_norm(%14, %v_param_17, %v_param_18, %v_param_19, %v_param_20, epsilon=0.001f) /* ty=(Tensor[(1, 64, 56, 56), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
  %16 = %15.0;
  %17 = clip(%16, a_min=0f, a_max=6f) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %18 = nn.conv2d(%17, %v_param_21, channels=128, kernel_size=[1, 1]) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %19 = nn.batch_norm(%18, %v_param_22, %v_param_23, %v_param_24, %v_param_25, epsilon=0.001f) /* ty=(Tensor[(1, 128, 56, 56), float32], Tensor[(128), float32], Tensor[(128), float32]) */;
  %20 = %19.0;
  %21 = clip(%20, a_min=0f, a_max=6f) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %22 = nn.conv2d(%21, %v_param_26, padding=[1, 1], groups=128, channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %23 = nn.batch_norm(%22, %v_param_27, %v_param_28, %v_param_29, %v_param_30, epsilon=0.001f) /* ty=(Tensor[(1, 128, 56, 56), float32], Tensor[(128), float32], Tensor[(128), float32]) */;
  %24 = %23.0;
  %25 = clip(%24, a_min=0f, a_max=6f) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %26 = nn.conv2d(%25, %v_param_31, channels=128, kernel_size=[1, 1]) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %27 = nn.batch_norm(%26, %v_param_32, %v_param_33, %v_param_34, %v_param_35, epsilon=0.001f) /* ty=(Tensor[(1, 128, 56, 56), float32], Tensor[(128), float32], Tensor[(128), float32]) */;
  %28 = %27.0;
  %29 = clip(%28, a_min=0f, a_max=6f) /* ty=Tensor[(1, 128, 56, 56), float32] */;
  %30 = nn.pad(%29, pad_width=[[0, 0], [0, 0], [0, 1], [0, 1]]) /* ty=Tensor[(1, 128, 57, 57), float32] */;
  %31 = nn.conv2d(%30, %v_param_36, strides=[2, 2], groups=128, channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %32 = nn.batch_norm(%31, %v_param_37, %v_param_38, %v_param_39, %v_param_40, epsilon=0.001f) /* ty=(Tensor[(1, 128, 28, 28), float32], Tensor[(128), float32], Tensor[(128), float32]) */;
  %33 = %32.0;
  %34 = clip(%33, a_min=0f, a_max=6f) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %35 = nn.conv2d(%34, %v_param_41, channels=256, kernel_size=[1, 1]) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %36 = nn.batch_norm(%35, %v_param_42, %v_param_43, %v_param_44, %v_param_45, epsilon=0.001f) /* ty=(Tensor[(1, 256, 28, 28), float32], Tensor[(256), float32], Tensor[(256), float32]) */;
  %37 = %36.0;
  %38 = clip(%37, a_min=0f, a_max=6f) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %39 = nn.conv2d(%38, %v_param_46, padding=[1, 1], groups=256, channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %40 = nn.batch_norm(%39, %v_param_47, %v_param_48, %v_param_49, %v_param_50, epsilon=0.001f) /* ty=(Tensor[(1, 256, 28, 28), float32], Tensor[(256), float32], Tensor[(256), float32]) */;
  %41 = %40.0;
  %42 = clip(%41, a_min=0f, a_max=6f) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %43 = nn.conv2d(%42, %v_param_51, channels=256, kernel_size=[1, 1]) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %44 = nn.batch_norm(%43, %v_param_52, %v_param_53, %v_param_54, %v_param_55, epsilon=0.001f) /* ty=(Tensor[(1, 256, 28, 28), float32], Tensor[(256), float32], Tensor[(256), float32]) */;
  %45 = %44.0;
  %46 = clip(%45, a_min=0f, a_max=6f) /* ty=Tensor[(1, 256, 28, 28), float32] */;
  %47 = nn.pad(%46, pad_width=[[0, 0], [0, 0], [0, 1], [0, 1]]) /* ty=Tensor[(1, 256, 29, 29), float32] */;
  %48 = nn.conv2d(%47, %v_param_56, strides=[2, 2], groups=256, channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %49 = nn.batch_norm(%48, %v_param_57, %v_param_58, %v_param_59, %v_param_60, epsilon=0.001f) /* ty=(Tensor[(1, 256, 14, 14), float32], Tensor[(256), float32], Tensor[(256), float32]) */;
  %50 = %49.0;
  %51 = clip(%50, a_min=0f, a_max=6f) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %52 = nn.conv2d(%51, %v_param_61, channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %53 = nn.batch_norm(%52, %v_param_62, %v_param_63, %v_param_64, %v_param_65, epsilon=0.001f) /* ty=(Tensor[(1, 512, 14, 14), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %54 = %53.0;
  %55 = clip(%54, a_min=0f, a_max=6f) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %56 = nn.conv2d(%55, %v_param_66, padding=[1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %57 = nn.batch_norm(%56, %v_param_67, %v_param_68, %v_param_69, %v_param_70, epsilon=0.001f) /* ty=(Tensor[(1, 512, 14, 14), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %58 = %57.0;
  %59 = clip(%58, a_min=0f, a_max=6f) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %60 = nn.conv2d(%59, %v_param_71, channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %61 = nn.batch_norm(%60, %v_param_72, %v_param_73, %v_param_74, %v_param_75, epsilon=0.001f) /* ty=(Tensor[(1, 512, 14, 14), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %62 = %61.0;
  %63 = clip(%62, a_min=0f, a_max=6f) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %64 = nn.conv2d(%63, %v_param_76, padding=[1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %65 = nn.batch_norm(%64, %v_param_77, %v_param_78, %v_param_79, %v_param_80, epsilon=0.001f) /* ty=(Tensor[(1, 512, 14, 14), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %66 = %65.0;
  %67 = clip(%66, a_min=0f, a_max=6f) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %68 = nn.conv2d(%67, %v_param_81, channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %69 = nn.batch_norm(%68, %v_param_82, %v_param_83, %v_param_84, %v_param_85, epsilon=0.001f) /* ty=(Tensor[(1, 512, 14, 14), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %70 = %69.0;
  %71 = clip(%70, a_min=0f, a_max=6f) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %72 = nn.conv2d(%71, %v_param_86, padding=[1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %73 = nn.batch_norm(%72, %v_param_87, %v_param_88, %v_param_89, %v_param_90, epsilon=0.001f) /* ty=(Tensor[(1, 512, 14, 14), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %74 = %73.0;
  %75 = clip(%74, a_min=0f, a_max=6f) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %76 = nn.conv2d(%75, %v_param_91, channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %77 = nn.batch_norm(%76, %v_param_92, %v_param_93, %v_param_94, %v_param_95, epsilon=0.001f) /* ty=(Tensor[(1, 512, 14, 14), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %78 = %77.0;
  %79 = clip(%78, a_min=0f, a_max=6f) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %80 = nn.conv2d(%79, %v_param_96, padding=[1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %81 = nn.batch_norm(%80, %v_param_97, %v_param_98, %v_param_99, %v_param_100, epsilon=0.001f) /* ty=(Tensor[(1, 512, 14, 14), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %82 = %81.0;
  %83 = clip(%82, a_min=0f, a_max=6f) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %84 = nn.conv2d(%83, %v_param_101, channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %85 = nn.batch_norm(%84, %v_param_102, %v_param_103, %v_param_104, %v_param_105, epsilon=0.001f) /* ty=(Tensor[(1, 512, 14, 14), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %86 = %85.0;
  %87 = clip(%86, a_min=0f, a_max=6f) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %88 = nn.conv2d(%87, %v_param_106, padding=[1, 1], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %89 = nn.batch_norm(%88, %v_param_107, %v_param_108, %v_param_109, %v_param_110, epsilon=0.001f) /* ty=(Tensor[(1, 512, 14, 14), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %90 = %89.0;
  %91 = clip(%90, a_min=0f, a_max=6f) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %92 = nn.conv2d(%91, %v_param_111, channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %93 = nn.batch_norm(%92, %v_param_112, %v_param_113, %v_param_114, %v_param_115, epsilon=0.001f) /* ty=(Tensor[(1, 512, 14, 14), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %94 = %93.0;
  %95 = clip(%94, a_min=0f, a_max=6f) /* ty=Tensor[(1, 512, 14, 14), float32] */;
  %96 = nn.pad(%95, pad_width=[[0, 0], [0, 0], [0, 1], [0, 1]]) /* ty=Tensor[(1, 512, 15, 15), float32] */;
  %97 = nn.conv2d(%96, %v_param_116, strides=[2, 2], groups=512, channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %98 = nn.batch_norm(%97, %v_param_117, %v_param_118, %v_param_119, %v_param_120, epsilon=0.001f) /* ty=(Tensor[(1, 512, 7, 7), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %99 = %98.0;
  %100 = clip(%99, a_min=0f, a_max=6f) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %101 = nn.conv2d(%100, %v_param_121, channels=1024, kernel_size=[1, 1]) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %102 = nn.batch_norm(%101, %v_param_122, %v_param_123, %v_param_124, %v_param_125, epsilon=0.001f) /* ty=(Tensor[(1, 1024, 7, 7), float32], Tensor[(1024), float32], Tensor[(1024), float32]) */;
  %103 = %102.0;
  %104 = clip(%103, a_min=0f, a_max=6f) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %105 = nn.conv2d(%104, %v_param_126, padding=[1, 1], groups=1024, channels=1024, kernel_size=[3, 3]) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %106 = nn.batch_norm(%105, %v_param_127, %v_param_128, %v_param_129, %v_param_130, epsilon=0.001f) /* ty=(Tensor[(1, 1024, 7, 7), float32], Tensor[(1024), float32], Tensor[(1024), float32]) */;
  %107 = %106.0;
  %108 = clip(%107, a_min=0f, a_max=6f) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %109 = nn.conv2d(%108, %v_param_131, channels=1024, kernel_size=[1, 1]) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %110 = nn.batch_norm(%109, %v_param_132, %v_param_133, %v_param_134, %v_param_135, epsilon=0.001f) /* ty=(Tensor[(1, 1024, 7, 7), float32], Tensor[(1024), float32], Tensor[(1024), float32]) */;
  %111 = %110.0;
  %112 = clip(%111, a_min=0f, a_max=6f) /* ty=Tensor[(1, 1024, 7, 7), float32] */;
  %113 = nn.global_avg_pool2d(%112) /* ty=Tensor[(1, 1024, 1, 1), float32] */;
  %114 = transpose(%113, axes=[0, 2, 3, 1]) /* ty=Tensor[(1, 1, 1, 1024), float32] */;
  %115 = nn.batch_flatten(%114) /* ty=Tensor[(1, 1024), float32] */;
  %116 = reshape(%115, newshape=[-1, 1024, 1, 1]) /* ty=Tensor[(1, 1024, 1, 1), float32] */;
  %117 = nn.conv2d(%116, %v_param_136, channels=1000, kernel_size=[1, 1]) /* ty=Tensor[(1, 1000, 1, 1), float32] */;
  %118 = nn.bias_add(%117, %v_param_137) /* ty=Tensor[(1, 1000, 1, 1), float32] */;
  %119 = reshape(%118, newshape=[-1, 1000]) /* ty=Tensor[(1, 1000), float32] */;
  nn.softmax(%119, axis=1) /* ty=Tensor[(1, 1000), float32] */
}
"""
hcl.init(hcl.Float())
def get_hcl_output(xs, dtype='float32'):
        shape_dict = {name: x.shape for (name, x) in zip(keras_model.input_names, xs)}
        return relay_parser.get_relay_model(keras_model, shape_dict, 'keras')
def to_channels_first(arr):
        if(len(arr.shape)>1):
                return arr.transpose([0, -1] + list(range(1, arr.ndim - 1)))
        else:
                return arr
keras_model = keras.applications.MobileNet(include_top=True, weights='imagenet',
        input_shape=(224, 224, 3), classes=1000)
in_shapes = []
for layer in keras_model._input_layers:
        in_shapes.append(tuple(dim.value if dim.value is not None else 1 for dim in layer.input.shape))
xs = [np.random.uniform(size=shape, low=1, high=10).astype('float32') for shape in in_shapes]
inputs = [to_channels_first(x) for x in xs]
f,param = get_hcl_output(inputs,dtype='float32')
params=[]
input_1 = hcl.placeholder((1,3,224,224))
params.append(input_1)
for i in range(len(params)):
        pass
params.append(hcl.placeholder((32,3,3,3)))
params.append(hcl.placeholder((32,)))
params.append(hcl.placeholder((32,)))
params.append(hcl.placeholder((32,)))
params.append(hcl.placeholder((32,)))
params.append(hcl.placeholder((32,1,3,3)))
params.append(hcl.placeholder((32,)))
params.append(hcl.placeholder((32,)))
params.append(hcl.placeholder((32,)))
params.append(hcl.placeholder((32,)))
def mobilenet_debug(input_1,*params):
	p0 = hlib.nn.relay_pad(input_1, [[0, 0], [0, 0], [0, 1], [0, 1]])
	p1 = hlib.nn.conv2d(p0,params[0],strides=[2, 2], channels=32, kernel_size=[3, 3])
	p2 = hlib.nn.batch_norm(p1,params[1],params[2],params[3],params[4],epsilon=0.001)
	p3 = hlib.math.clip(p2[0],a_min=0.0,a_max=6.0)
	p4 = hlib.nn.conv2d(p3,params[5],strides=[2, 2], channels=32, kernel_size=[3, 3])
	#p5 = hlib.nn.batch_norm(p4,params[6],params[7],params[8],params[9],epsilon=0.001)
	#p6 = hlib.math.clip(p5[0],a_min=0.0,a_max=6.0)
	return p4
for par in param:
    par = hcl.asarray(par)
s=hcl.create_schedule(params,mobilenet_debug)
func = hcl.build(s)
#out = keras_model.predict(xs)
out_layer_name = 'conv_dw_1'
input_1 = hcl.asarray(inputs[0])
out = hcl.placeholder((1,32,112,112))
out = hcl.asarray(np.zeros((1,32,112,112)).astype(float))
intermediate_output = Model(inputs=keras_model.input,
outputs=keras_model.get_layer(out_layer_name).output)
k_out = intermediate_output.predict(xs)
func(input_1,*param[0:10],out)
shape = out.shape
h_out = out.asnumpy()
#h_out = np.reshape(h_out,(shape[0],shape[2],shape[3],shape[1]))
h_out = np.transpose(h_out,[0,2,3,1])
print(h_out)
print(k_out)
#print(keras_model.summary())
tst.assert_almost_equal(h_out,k_out,10**-6)
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0
_________________________________________________________________
conv1_pad (ZeroPadding2D)    (None, 225, 225, 3)       0
_________________________________________________________________
conv1 (Conv2D)               (None, 112, 112, 32)      864
_________________________________________________________________
conv1_bn (BatchNormalization (None, 112, 112, 32)      128
_________________________________________________________________
conv1_relu (ReLU)            (None, 112, 112, 32)      0
_________________________________________________________________
conv_dw_1 (DepthwiseConv2D)  (None, 112, 112, 32)      288
_________________________________________________________________
conv_dw_1_bn (BatchNormaliza (None, 112, 112, 32)      128
_________________________________________________________________
conv_dw_1_relu (ReLU)        (None, 112, 112, 32)      0
_________________________________________________________________
conv_pw_1 (Conv2D)           (None, 112, 112, 64)      2048
_________________________________________________________________
conv_pw_1_bn (BatchNormaliza (None, 112, 112, 64)      256
_________________________________________________________________
conv_pw_1_relu (ReLU)        (None, 112, 112, 64)      0
_________________________________________________________________
conv_pad_2 (ZeroPadding2D)   (None, 113, 113, 64)      0
_________________________________________________________________
conv_dw_2 (DepthwiseConv2D)  (None, 56, 56, 64)        576
_________________________________________________________________
conv_dw_2_bn (BatchNormaliza (None, 56, 56, 64)        256
_________________________________________________________________
conv_dw_2_relu (ReLU)        (None, 56, 56, 64)        0
_________________________________________________________________
conv_pw_2 (Conv2D)           (None, 56, 56, 128)       8192
_________________________________________________________________
conv_pw_2_bn (BatchNormaliza (None, 56, 56, 128)       512
_________________________________________________________________
conv_pw_2_relu (ReLU)        (None, 56, 56, 128)       0
_________________________________________________________________
conv_dw_3 (DepthwiseConv2D)  (None, 56, 56, 128)       1152
_________________________________________________________________
conv_dw_3_bn (BatchNormaliza (None, 56, 56, 128)       512
_________________________________________________________________
conv_dw_3_relu (ReLU)        (None, 56, 56, 128)       0
_________________________________________________________________
conv_pw_3 (Conv2D)           (None, 56, 56, 128)       16384
_________________________________________________________________
conv_pw_3_bn (BatchNormaliza (None, 56, 56, 128)       512
_________________________________________________________________
conv_pw_3_relu (ReLU)        (None, 56, 56, 128)       0
_________________________________________________________________
conv_pad_4 (ZeroPadding2D)   (None, 57, 57, 128)       0
_________________________________________________________________
conv_dw_4 (DepthwiseConv2D)  (None, 28, 28, 128)       1152
_________________________________________________________________
conv_dw_4_bn (BatchNormaliza (None, 28, 28, 128)       512
_________________________________________________________________
conv_dw_4_relu (ReLU)        (None, 28, 28, 128)       0
_________________________________________________________________
conv_pw_4 (Conv2D)           (None, 28, 28, 256)       32768
_________________________________________________________________
conv_pw_4_bn (BatchNormaliza (None, 28, 28, 256)       1024
_________________________________________________________________
conv_pw_4_relu (ReLU)        (None, 28, 28, 256)       0
_________________________________________________________________
conv_dw_5 (DepthwiseConv2D)  (None, 28, 28, 256)       2304
_________________________________________________________________
conv_dw_5_bn (BatchNormaliza (None, 28, 28, 256)       1024
_________________________________________________________________
conv_dw_5_relu (ReLU)        (None, 28, 28, 256)       0
_________________________________________________________________
conv_pw_5 (Conv2D)           (None, 28, 28, 256)       65536
_________________________________________________________________
conv_pw_5_bn (BatchNormaliza (None, 28, 28, 256)       1024
_________________________________________________________________
conv_pw_5_relu (ReLU)        (None, 28, 28, 256)       0
_________________________________________________________________
conv_pad_6 (ZeroPadding2D)   (None, 29, 29, 256)       0
_________________________________________________________________
conv_dw_6 (DepthwiseConv2D)  (None, 14, 14, 256)       2304
_________________________________________________________________
conv_dw_6_bn (BatchNormaliza (None, 14, 14, 256)       1024
_________________________________________________________________
conv_dw_6_relu (ReLU)        (None, 14, 14, 256)       0
_________________________________________________________________
conv_pw_6 (Conv2D)           (None, 14, 14, 512)       131072
_________________________________________________________________
conv_pw_6_bn (BatchNormaliza (None, 14, 14, 512)       2048
_________________________________________________________________
conv_pw_6_relu (ReLU)        (None, 14, 14, 512)       0
_________________________________________________________________
conv_dw_7 (DepthwiseConv2D)  (None, 14, 14, 512)       4608
_________________________________________________________________
conv_dw_7_bn (BatchNormaliza (None, 14, 14, 512)       2048
_________________________________________________________________
conv_dw_7_relu (ReLU)        (None, 14, 14, 512)       0
_________________________________________________________________
conv_pw_7 (Conv2D)           (None, 14, 14, 512)       262144
_________________________________________________________________
conv_pw_7_bn (BatchNormaliza (None, 14, 14, 512)       2048
_________________________________________________________________
conv_pw_7_relu (ReLU)        (None, 14, 14, 512)       0
_________________________________________________________________
conv_dw_8 (DepthwiseConv2D)  (None, 14, 14, 512)       4608
_________________________________________________________________
conv_dw_8_bn (BatchNormaliza (None, 14, 14, 512)       2048
_________________________________________________________________
conv_dw_8_relu (ReLU)        (None, 14, 14, 512)       0
_________________________________________________________________
conv_pw_8 (Conv2D)           (None, 14, 14, 512)       262144
_________________________________________________________________
conv_pw_8_bn (BatchNormaliza (None, 14, 14, 512)       2048
_________________________________________________________________
conv_pw_8_relu (ReLU)        (None, 14, 14, 512)       0
_________________________________________________________________
conv_dw_9 (DepthwiseConv2D)  (None, 14, 14, 512)       4608
_________________________________________________________________
conv_dw_9_bn (BatchNormaliza (None, 14, 14, 512)       2048
_________________________________________________________________
conv_dw_9_relu (ReLU)        (None, 14, 14, 512)       0
_________________________________________________________________
conv_pw_9 (Conv2D)           (None, 14, 14, 512)       262144
_________________________________________________________________
conv_pw_9_bn (BatchNormaliza (None, 14, 14, 512)       2048
_________________________________________________________________
conv_pw_9_relu (ReLU)        (None, 14, 14, 512)       0
_________________________________________________________________
conv_dw_10 (DepthwiseConv2D) (None, 14, 14, 512)       4608
_________________________________________________________________
conv_dw_10_bn (BatchNormaliz (None, 14, 14, 512)       2048
_________________________________________________________________
conv_dw_10_relu (ReLU)       (None, 14, 14, 512)       0
_________________________________________________________________
conv_pw_10 (Conv2D)          (None, 14, 14, 512)       262144
_________________________________________________________________
conv_pw_10_bn (BatchNormaliz (None, 14, 14, 512)       2048
_________________________________________________________________
conv_pw_10_relu (ReLU)       (None, 14, 14, 512)       0
_________________________________________________________________
conv_dw_11 (DepthwiseConv2D) (None, 14, 14, 512)       4608
_________________________________________________________________
conv_dw_11_bn (BatchNormaliz (None, 14, 14, 512)       2048
_________________________________________________________________
conv_dw_11_relu (ReLU)       (None, 14, 14, 512)       0
_________________________________________________________________
conv_pw_11 (Conv2D)          (None, 14, 14, 512)       262144
_________________________________________________________________
conv_pw_11_bn (BatchNormaliz (None, 14, 14, 512)       2048
_________________________________________________________________
conv_pw_11_relu (ReLU)       (None, 14, 14, 512)       0
_________________________________________________________________
conv_pad_12 (ZeroPadding2D)  (None, 15, 15, 512)       0
_________________________________________________________________
conv_dw_12 (DepthwiseConv2D) (None, 7, 7, 512)         4608
_________________________________________________________________
conv_dw_12_bn (BatchNormaliz (None, 7, 7, 512)         2048
_________________________________________________________________
conv_dw_12_relu (ReLU)       (None, 7, 7, 512)         0
_________________________________________________________________
conv_pw_12 (Conv2D)          (None, 7, 7, 1024)        524288
_________________________________________________________________
conv_pw_12_bn (BatchNormaliz (None, 7, 7, 1024)        4096
_________________________________________________________________
conv_pw_12_relu (ReLU)       (None, 7, 7, 1024)        0
_________________________________________________________________
conv_dw_13 (DepthwiseConv2D) (None, 7, 7, 1024)        9216
_________________________________________________________________
conv_dw_13_bn (BatchNormaliz (None, 7, 7, 1024)        4096
_________________________________________________________________
conv_dw_13_relu (ReLU)       (None, 7, 7, 1024)        0
_________________________________________________________________
conv_pw_13 (Conv2D)          (None, 7, 7, 1024)        1048576
_________________________________________________________________
conv_pw_13_bn (BatchNormaliz (None, 7, 7, 1024)        4096
_________________________________________________________________
conv_pw_13_relu (ReLU)       (None, 7, 7, 1024)        0
_________________________________________________________________
global_average_pooling2d_1 ( (None, 1024)              0
_________________________________________________________________
reshape_1 (Reshape)          (None, 1, 1, 1024)        0
_________________________________________________________________
dropout (Dropout)            (None, 1, 1, 1024)        0
_________________________________________________________________
conv_preds (Conv2D)          (None, 1, 1, 1000)        1025000
_________________________________________________________________
reshape_2 (Reshape)          (None, 1000)              0
_________________________________________________________________
act_softmax (Activation)     (None, 1000)              0
=================================================================
Total params: 4,253,864
Trainable params: 4,231,976
Non-trainable params: 21,888
_________________________________________________________________
"""