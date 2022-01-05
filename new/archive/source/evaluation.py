#!/usr/bin/env python3

# Evaluation across 4 metrics

import numpy as np

# def evaluate(predicted, gold):
#   FFDAvg_mae = np.abs(predicted['FFDAvg'] - gold['FFDAvg']).mean() 
#   FFDStd_mae = np.abs(predicted['FFDStd'] - gold['FFDStd']).mean()
#   TRTAvg_mae = np.abs(predicted['TRTAvg'] - gold['TRTAvg']).mean()
#   TRTStd_mae = np.abs(predicted['TRTStd'] - gold['TRTStd']).mean()
#   overall_mae = (FFDAvg_mae+FFDStd_mae+TRTAvg_mae+TRTStd_mae) / 4
#   return FFDAvg_mae,FFDStd_mae,TRTAvg_mae,TRTStd_mae,overall_mae

def evaluate(predicted, gold):
  mae = np.abs(predicted - gold).mean() 
  return mae
  