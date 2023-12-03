import numpy as np

from parameters import *

thymio_data = [{'ground': [177, 496], 'sensor': [177, 496], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [178, 495], 'sensor': [178, 495], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [177, 495], 'sensor': [177, 495], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [177, 496], 'sensor': [177, 496], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [177, 495], 'sensor': [177, 495], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [178, 496], 'sensor': [178, 496], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [178, 495], 'sensor': [178, 495], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [177, 496], 'sensor': [177, 496], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [177, 496], 'sensor': [177, 496], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [178, 495], 'sensor': [178, 495], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [177, 495], 'sensor': [177, 495], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [177, 497], 'sensor': [177, 497], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [177, 495], 'sensor': [177, 495], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [177, 495], 'sensor': [177, 495], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [178, 496], 'sensor': [178, 496], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [180, 546], 'sensor': [180, 546], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [177, 454], 'sensor': [177, 454], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [97, 69], 'sensor': [97, 69], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [39, 41], 'sensor': [39, 41], 'left_speed': 0, 'right_speed': 0},
               {'ground': [82, 47], 'sensor': [82, 47], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [34, 29], 'sensor': [34, 29], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [24, 9], 'sensor': [24, 9], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [899, 447], 'sensor': [899, 447], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [967, 631], 'sensor': [967, 631], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [972, 682], 'sensor': [972, 682], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [971, 675], 'sensor': [971, 675], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [970, 651], 'sensor': [970, 651], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [971, 642], 'sensor': [971, 642], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [970, 628], 'sensor': [970, 628], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [970, 628], 'sensor': [970, 628], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [970, 628], 'sensor': [970, 628], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [970, 627], 'sensor': [970, 627], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [970, 611], 'sensor': [970, 611], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [971, 604], 'sensor': [971, 604], 'left_speed': 0, 'right_speed': 0},
               {'ground': [971, 598], 'sensor': [971, 598], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [972, 642], 'sensor': [972, 642], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [973, 714], 'sensor': [973, 714], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [973, 704], 'sensor': [973, 704], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [973, 702], 'sensor': [973, 702], 'left_speed': 0, 'right_speed': 0},
               {'ground': [972, 700], 'sensor': [972, 700], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [973, 701], 'sensor': [973, 701], 'left_speed': 0, 'right_speed': 0},
               {'ground': [974, 707], 'sensor': [974, 707], 'left_speed': 0, 'right_speed': 0},
               {'ground': [974, 711], 'sensor': [974, 711], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [974, 707], 'sensor': [974, 707], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [972, 702], 'sensor': [972, 702], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [973, 686], 'sensor': [973, 686], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [972, 662], 'sensor': [972, 662], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [968, 602], 'sensor': [968, 602], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [954, 539], 'sensor': [954, 539], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [833, 464], 'sensor': [833, 464], 'left_speed': 0, 'right_speed': 65535},
               {'ground': [826, 460], 'sensor': [826, 460], 'left_speed': 2, 'right_speed': 65535},
               {'ground': [511, 244], 'sensor': [511, 244], 'left_speed': 39, 'right_speed': 31},
               {'ground': [306, 131], 'sensor': [306, 131], 'left_speed': 49, 'right_speed': 35},
               {'ground': [194, 87], 'sensor': [194, 87], 'left_speed': 58, 'right_speed': 57},
               {'ground': [174, 84], 'sensor': [174, 84], 'left_speed': 64, 'right_speed': 42},
               {'ground': [174, 84], 'sensor': [174, 84], 'left_speed': 38, 'right_speed': 32},
               {'ground': [174, 84], 'sensor': [174, 84], 'left_speed': 38, 'right_speed': 32},
               {'ground': [171, 84], 'sensor': [171, 84], 'left_speed': 58, 'right_speed': 30},
               {'ground': [169, 81], 'sensor': [169, 81], 'left_speed': 49, 'right_speed': 56},
               {'ground': [167, 80], 'sensor': [167, 80], 'left_speed': 42, 'right_speed': 55},
               {'ground': [167, 81], 'sensor': [167, 81], 'left_speed': 46, 'right_speed': 35},
               {'ground': [168, 80], 'sensor': [168, 80], 'left_speed': 59, 'right_speed': 57},
               {'ground': [172, 81], 'sensor': [172, 81], 'left_speed': 63, 'right_speed': 54},
               {'ground': [173, 84], 'sensor': [173, 84], 'left_speed': 45, 'right_speed': 56},
               {'ground': [171, 84], 'sensor': [171, 84], 'left_speed': 60, 'right_speed': 55},
               {'ground': [171, 84], 'sensor': [171, 84], 'left_speed': 60, 'right_speed': 55},
               {'ground': [165, 84], 'sensor': [165, 84], 'left_speed': 54, 'right_speed': 36},
               {'ground': [166, 86], 'sensor': [166, 86], 'left_speed': 57, 'right_speed': 43},
               {'ground': [166, 86], 'sensor': [166, 86], 'left_speed': 41, 'right_speed': 43},
               {'ground': [168, 87], 'sensor': [168, 87], 'left_speed': 59, 'right_speed': 47},
               {'ground': [175, 96], 'sensor': [175, 96], 'left_speed': 42, 'right_speed': 48},
               {'ground': [280, 157], 'sensor': [280, 157], 'left_speed': 61, 'right_speed': 50},
               {'ground': [503, 262], 'sensor': [503, 262], 'left_speed': 60, 'right_speed': 49},
               {'ground': [658, 355], 'sensor': [658, 355], 'left_speed': 46, 'right_speed': 36},
               {'ground': [639, 345], 'sensor': [639, 345], 'left_speed': 49, 'right_speed': 57},
               {'ground': [639, 345], 'sensor': [639, 345], 'left_speed': 49, 'right_speed': 57},
               {'ground': [501, 257], 'sensor': [501, 257], 'left_speed': 42, 'right_speed': 51},
               {'ground': [187, 88], 'sensor': [187, 88], 'left_speed': 45, 'right_speed': 60},
               {'ground': [166, 85], 'sensor': [166, 85], 'left_speed': 53, 'right_speed': 56},
               {'ground': [166, 84], 'sensor': [166, 84], 'left_speed': 60, 'right_speed': 55},
               {'ground': [168, 84], 'sensor': [168, 84], 'left_speed': 50, 'right_speed': 53},
               {'ground': [169, 85], 'sensor': [169, 85], 'left_speed': 57, 'right_speed': 47},
               {'ground': [167, 87], 'sensor': [167, 87], 'left_speed': 49, 'right_speed': 41},
               {'ground': [169, 86], 'sensor': [169, 86], 'left_speed': 47, 'right_speed': 45},
               {'ground': [167, 84], 'sensor': [167, 84], 'left_speed': 66, 'right_speed': 33},
               {'ground': [164, 85], 'sensor': [164, 85], 'left_speed': 56, 'right_speed': 62},
               {'ground': [162, 85], 'sensor': [162, 85], 'left_speed': 43, 'right_speed': 54},
               {'ground': [163, 85], 'sensor': [163, 85], 'left_speed': 54, 'right_speed': 58},
               {'ground': [163, 85], 'sensor': [163, 85], 'left_speed': 41, 'right_speed': 54},
               {'ground': [160, 84], 'sensor': [160, 84], 'left_speed': 59, 'right_speed': 56},
               {'ground': [156, 82], 'sensor': [156, 82], 'left_speed': 52, 'right_speed': 60},
               {'ground': [157, 82], 'sensor': [157, 82], 'left_speed': 64, 'right_speed': 48},
               {'ground': [162, 83], 'sensor': [162, 83], 'left_speed': 64, 'right_speed': 56},
               {'ground': [166, 82], 'sensor': [166, 82], 'left_speed': 47, 'right_speed': 55},
               {'ground': [165, 91], 'sensor': [165, 91], 'left_speed': 41, 'right_speed': 44},
               {'ground': [254, 156], 'sensor': [254, 156], 'left_speed': 67, 'right_speed': 59},
               {'ground': [477, 279], 'sensor': [477, 279], 'left_speed': 53, 'right_speed': 57},
               {'ground': [678, 391], 'sensor': [678, 391], 'left_speed': 57, 'right_speed': 59},
               {'ground': [762, 426], 'sensor': [762, 426], 'left_speed': 67, 'right_speed': 57},
               {'ground': [738, 394], 'sensor': [738, 394], 'left_speed': 41, 'right_speed': 50},
               {'ground': [649, 332], 'sensor': [649, 332], 'left_speed': 60, 'right_speed': 56},
               {'ground': [596, 319], 'sensor': [596, 319], 'left_speed': 65, 'right_speed': 59},
               {'ground': [593, 321], 'sensor': [593, 321], 'left_speed': 47, 'right_speed': 48},
               {'ground': [596, 325], 'sensor': [596, 325], 'left_speed': 58, 'right_speed': 33},
               {'ground': [601, 330], 'sensor': [601, 330], 'left_speed': 58, 'right_speed': 55},
               {'ground': [605, 331], 'sensor': [605, 331], 'left_speed': 48, 'right_speed': 54},
               {'ground': [608, 331], 'sensor': [608, 331], 'left_speed': 42, 'right_speed': 43},
               {'ground': [607, 330], 'sensor': [607, 330], 'left_speed': 64, 'right_speed': 55},
               {'ground': [605, 329], 'sensor': [605, 329], 'left_speed': 46, 'right_speed': 42},
               {'ground': [605, 329], 'sensor': [605, 329], 'left_speed': 55, 'right_speed': 61},
               {'ground': [604, 330], 'sensor': [604, 330], 'left_speed': 52, 'right_speed': 56},
               {'ground': [604, 330], 'sensor': [604, 330], 'left_speed': 43, 'right_speed': 47},
               {'ground': [607, 331], 'sensor': [607, 331], 'left_speed': 64, 'right_speed': 54},
               {'ground': [604, 332], 'sensor': [604, 332], 'left_speed': 63, 'right_speed': 57},
               {'ground': [600, 334], 'sensor': [600, 334], 'left_speed': 56, 'right_speed': 61},
               {'ground': [598, 335], 'sensor': [598, 335], 'left_speed': 64, 'right_speed': 57},
               {'ground': [594, 337], 'sensor': [594, 337], 'left_speed': 60, 'right_speed': 51},
               {'ground': [592, 339], 'sensor': [592, 339], 'left_speed': 49, 'right_speed': 48},
               {'ground': [593, 350], 'sensor': [593, 350], 'left_speed': 57, 'right_speed': 50},
               {'ground': [642, 404], 'sensor': [642, 404], 'left_speed': 58, 'right_speed': 45},
               {'ground': [776, 487], 'sensor': [776, 487], 'left_speed': 42, 'right_speed': 56},
               {'ground': [776, 487], 'sensor': [776, 487], 'left_speed': 42, 'right_speed': 56},
               {'ground': [863, 531], 'sensor': [863, 531], 'left_speed': 65, 'right_speed': 53},
               {'ground': [760, 410], 'sensor': [760, 410], 'left_speed': 54, 'right_speed': 48},
               {'ground': [660, 363], 'sensor': [660, 363], 'left_speed': 43, 'right_speed': 49},
               {'ground': [617, 364], 'sensor': [617, 364], 'left_speed': 55, 'right_speed': 46},
               {'ground': [617, 364], 'sensor': [617, 364], 'left_speed': 55, 'right_speed': 46},
               {'ground': [599, 356], 'sensor': [599, 356], 'left_speed': 64, 'right_speed': 46},
               {'ground': [600, 348], 'sensor': [600, 348], 'left_speed': 48, 'right_speed': 46},
               {'ground': [604, 343], 'sensor': [604, 343], 'left_speed': 54, 'right_speed': 46},
               {'ground': [601, 340], 'sensor': [601, 340], 'left_speed': 63, 'right_speed': 46},
               {'ground': [598, 339], 'sensor': [598, 339], 'left_speed': 46, 'right_speed': 49},
               {'ground': [594, 340], 'sensor': [594, 340], 'left_speed': 53, 'right_speed': 48},
               {'ground': [590, 343], 'sensor': [590, 343], 'left_speed': 64, 'right_speed': 48},
               {'ground': [592, 347], 'sensor': [592, 347], 'left_speed': 47, 'right_speed': 54},
               {'ground': [594, 348], 'sensor': [594, 348], 'left_speed': 59, 'right_speed': 51},
               {'ground': [595, 352], 'sensor': [595, 352], 'left_speed': 53, 'right_speed': 49},
               {'ground': [598, 352], 'sensor': [598, 352], 'left_speed': 48, 'right_speed': 50},
               {'ground': [601, 352], 'sensor': [601, 352], 'left_speed': 59, 'right_speed': 46},
               {'ground': [604, 354], 'sensor': [604, 354], 'left_speed': 64, 'right_speed': 47},
               {'ground': [605, 354], 'sensor': [605, 354], 'left_speed': 45, 'right_speed': 45},
               {'ground': [617, 386], 'sensor': [617, 386], 'left_speed': 52, 'right_speed': 52},
               {'ground': [702, 466], 'sensor': [702, 466], 'left_speed': 65, 'right_speed': 51},
               {'ground': [831, 497], 'sensor': [831, 497], 'left_speed': 48, 'right_speed': 47},
               {'ground': [774, 398], 'sensor': [774, 398], 'left_speed': 56, 'right_speed': 50},
               {'ground': [603, 234], 'sensor': [603, 234], 'left_speed': 56, 'right_speed': 54},
               {'ground': [382, 118], 'sensor': [382, 118], 'left_speed': 44, 'right_speed': 53},
               {'ground': [213, 97], 'sensor': [213, 97], 'left_speed': 53, 'right_speed': 53},
               {'ground': [184, 100], 'sensor': [184, 100], 'left_speed': 67, 'right_speed': 49},
               {'ground': [186, 99], 'sensor': [186, 99], 'left_speed': 45, 'right_speed': 44},
               {'ground': [186, 98], 'sensor': [186, 98], 'left_speed': 58, 'right_speed': 43},
               {'ground': [189, 98], 'sensor': [189, 98], 'left_speed': 54, 'right_speed': 45},
               {'ground': [190, 94], 'sensor': [190, 94], 'left_speed': 53, 'right_speed': 50},
               {'ground': [188, 91], 'sensor': [188, 91], 'left_speed': 63, 'right_speed': 50},
               {'ground': [184, 93], 'sensor': [184, 93], 'left_speed': 48, 'right_speed': 50},
               {'ground': [187, 92], 'sensor': [187, 92], 'left_speed': 56, 'right_speed': 49},
               {'ground': [189, 92], 'sensor': [189, 92], 'left_speed': 62, 'right_speed': 49},
               {'ground': [202, 94], 'sensor': [202, 94], 'left_speed': 46, 'right_speed': 46},
               {'ground': [172, 95], 'sensor': [172, 95], 'left_speed': 58, 'right_speed': 50},
               {'ground': [176, 78], 'sensor': [176, 78], 'left_speed': 47, 'right_speed': 49},
               {'ground': [177, 79], 'sensor': [177, 79], 'left_speed': 56, 'right_speed': 49},
               {'ground': [171, 72], 'sensor': [171, 72], 'left_speed': 55, 'right_speed': 48},
               {'ground': [159, 87], 'sensor': [159, 87], 'left_speed': 52, 'right_speed': 47},
               {'ground': [169, 165], 'sensor': [169, 165], 'left_speed': 50, 'right_speed': 45},
               {'ground': [304, 285], 'sensor': [304, 285], 'left_speed': 52, 'right_speed': 48},
               {'ground': [548, 374], 'sensor': [548, 374], 'left_speed': 55, 'right_speed': 49},
               {'ground': [663, 354], 'sensor': [663, 354], 'left_speed': 55, 'right_speed': 50},
               {'ground': [586, 206], 'sensor': [586, 206], 'left_speed': 52, 'right_speed': 51},
               {'ground': [403, 93], 'sensor': [403, 93], 'left_speed': 58, 'right_speed': 50},
               {'ground': [225, 84], 'sensor': [225, 84], 'left_speed': 59, 'right_speed': 47},
               {'ground': [181, 91], 'sensor': [181, 91], 'left_speed': 46, 'right_speed': 47},
               {'ground': [181, 91], 'sensor': [181, 91], 'left_speed': 46, 'right_speed': 47},
               {'ground': [179, 99], 'sensor': [179, 99], 'left_speed': 63, 'right_speed': 49},
               {'ground': [192, 98], 'sensor': [192, 98], 'left_speed': 53, 'right_speed': 53},
               {'ground': [194, 95], 'sensor': [194, 95], 'left_speed': 57, 'right_speed': 49},
               {'ground': [183, 91], 'sensor': [183, 91], 'left_speed': 56, 'right_speed': 45},
               {'ground': [176, 89], 'sensor': [176, 89], 'left_speed': 47, 'right_speed': 45},
               {'ground': [175, 90], 'sensor': [175, 90], 'left_speed': 53, 'right_speed': 47},
               {'ground': [178, 92], 'sensor': [178, 92], 'left_speed': 66, 'right_speed': 48},
               {'ground': [177, 94], 'sensor': [177, 94], 'left_speed': 49, 'right_speed': 48},
               {'ground': [180, 90], 'sensor': [180, 90], 'left_speed': 59, 'right_speed': 50},
               {'ground': [181, 88], 'sensor': [181, 88], 'left_speed': 58, 'right_speed': 47},
               {'ground': [179, 87], 'sensor': [179, 87], 'left_speed': 46, 'right_speed': 51},
               {'ground': [181, 90], 'sensor': [181, 90], 'left_speed': 59, 'right_speed': 51},
               {'ground': [189, 100], 'sensor': [189, 100], 'left_speed': 50, 'right_speed': 51},
               {'ground': [193, 153], 'sensor': [193, 153], 'left_speed': 56, 'right_speed': 48},
               {'ground': [317, 316], 'sensor': [317, 316], 'left_speed': 53, 'right_speed': 48},
               {'ground': [605, 412], 'sensor': [605, 412], 'left_speed': 47, 'right_speed': 48},
               {'ground': [605, 412], 'sensor': [605, 412], 'left_speed': 47, 'right_speed': 48},
               {'ground': [684, 335], 'sensor': [684, 335], 'left_speed': 60, 'right_speed': 49},
               {'ground': [575, 174], 'sensor': [575, 174], 'left_speed': 47, 'right_speed': 49},
               {'ground': [224, 86], 'sensor': [224, 86], 'left_speed': 53, 'right_speed': 49},
               {'ground': [180, 86], 'sensor': [180, 86], 'left_speed': 49, 'right_speed': 46},
               {'ground': [175, 85], 'sensor': [175, 85], 'left_speed': 65, 'right_speed': 49},
               {'ground': [175, 85], 'sensor': [175, 85], 'left_speed': 65, 'right_speed': 49},
               {'ground': [177, 86], 'sensor': [177, 86], 'left_speed': 48, 'right_speed': 47},
               {'ground': [182, 88], 'sensor': [182, 88], 'left_speed': 51, 'right_speed': 47},
               {'ground': [181, 87], 'sensor': [181, 87], 'left_speed': 62, 'right_speed': 50},
               {'ground': [176, 86], 'sensor': [176, 86], 'left_speed': 46, 'right_speed': 50},
               {'ground': [177, 89], 'sensor': [177, 89], 'left_speed': 65, 'right_speed': 45},
               {'ground': [183, 94], 'sensor': [183, 94], 'left_speed': 46, 'right_speed': 47},
               {'ground': [183, 94], 'sensor': [183, 94], 'left_speed': 46, 'right_speed': 47},
               {'ground': [189, 96], 'sensor': [189, 96], 'left_speed': 54, 'right_speed': 46},
               {'ground': [189, 97], 'sensor': [189, 97], 'left_speed': 42, 'right_speed': 45},
               {'ground': [188, 96], 'sensor': [188, 96], 'left_speed': 55, 'right_speed': 47},
               {'ground': [189, 97], 'sensor': [189, 97], 'left_speed': 45, 'right_speed': 49},
               {'ground': [190, 101], 'sensor': [190, 101], 'left_speed': 53, 'right_speed': 48},
               {'ground': [190, 101], 'sensor': [190, 101], 'left_speed': 53, 'right_speed': 48},
               {'ground': [241, 318], 'sensor': [241, 318], 'left_speed': 62, 'right_speed': 49},
               {'ground': [241, 318], 'sensor': [241, 318], 'left_speed': 62, 'right_speed': 49},
               {'ground': [770, 478], 'sensor': [770, 478], 'left_speed': 55, 'right_speed': 48},
               {'ground': [815, 401], 'sensor': [815, 401], 'left_speed': 65, 'right_speed': 53},
               {'ground': [753, 350], 'sensor': [753, 350], 'left_speed': 47, 'right_speed': 47},
               {'ground': [753, 350], 'sensor': [753, 350], 'left_speed': 47, 'right_speed': 47},
               {'ground': [658, 346], 'sensor': [658, 346], 'left_speed': 55, 'right_speed': 49},
               {'ground': [616, 344], 'sensor': [616, 344], 'left_speed': 57, 'right_speed': 48},
               {'ground': [612, 342], 'sensor': [612, 342], 'left_speed': 47, 'right_speed': 44},
               {'ground': [608, 344], 'sensor': [608, 344], 'left_speed': 62, 'right_speed': 46},
               {'ground': [606, 341], 'sensor': [606, 341], 'left_speed': 53, 'right_speed': 49},
               {'ground': [604, 342], 'sensor': [604, 342], 'left_speed': 60, 'right_speed': 49},
               {'ground': [604, 340], 'sensor': [604, 340], 'left_speed': 45, 'right_speed': 51},
               {'ground': [597, 339], 'sensor': [597, 339], 'left_speed': 62, 'right_speed': 48},
               {'ground': [592, 335], 'sensor': [592, 335], 'left_speed': 46, 'right_speed': 46},
               {'ground': [590, 332], 'sensor': [590, 332], 'left_speed': 52, 'right_speed': 53},
               {'ground': [589, 333], 'sensor': [589, 333], 'left_speed': 53, 'right_speed': 59},
               {'ground': [591, 333], 'sensor': [591, 333], 'left_speed': 64, 'right_speed': 65},
               {'ground': [592, 333], 'sensor': [592, 333], 'left_speed': 46, 'right_speed': 60},
               {'ground': [591, 333], 'sensor': [591, 333], 'left_speed': 55, 'right_speed': 49},
               {'ground': [594, 334], 'sensor': [594, 334], 'left_speed': 61, 'right_speed': 33},
               {'ground': [597, 360], 'sensor': [597, 360], 'left_speed': 48, 'right_speed': 56},
               {'ground': [601, 431], 'sensor': [601, 431], 'left_speed': 61, 'right_speed': 57},
               {'ground': [639, 549], 'sensor': [639, 549], 'left_speed': 61, 'right_speed': 59},
               {'ground': [776, 671], 'sensor': [776, 671], 'left_speed': 56, 'right_speed': 58},
               {'ground': [943, 750], 'sensor': [943, 750], 'left_speed': 60, 'right_speed': 60},
               {'ground': [967, 768], 'sensor': [967, 768], 'left_speed': 46, 'right_speed': 58},
               {'ground': [972, 770], 'sensor': [972, 770], 'left_speed': 49, 'right_speed': 51},
               {'ground': [974, 770], 'sensor': [974, 770], 'left_speed': 49, 'right_speed': 54},
               {'ground': [973, 769], 'sensor': [973, 769], 'left_speed': 64, 'right_speed': 58},
               {'ground': [973, 770], 'sensor': [973, 770], 'left_speed': 41, 'right_speed': 44},
               {'ground': [974, 768], 'sensor': [974, 768], 'left_speed': 61, 'right_speed': 55},
               {'ground': [973, 757], 'sensor': [973, 757], 'left_speed': 56, 'right_speed': 47},
               {'ground': [974, 736], 'sensor': [974, 736], 'left_speed': 43, 'right_speed': 40},
               {'ground': [974, 708], 'sensor': [974, 708], 'left_speed': 55, 'right_speed': 42},
               {'ground': [972, 720], 'sensor': [972, 720], 'left_speed': 47, 'right_speed': 49},
               {'ground': [970, 716], 'sensor': [970, 716], 'left_speed': 48, 'right_speed': 42},
               {'ground': [972, 708], 'sensor': [972, 708], 'left_speed': 50, 'right_speed': 35},
               {'ground': [970, 702], 'sensor': [970, 702], 'left_speed': 40, 'right_speed': 60},
               {'ground': [970, 702], 'sensor': [970, 702], 'left_speed': 43, 'right_speed': 57},
               {'ground': [969, 694], 'sensor': [969, 694], 'left_speed': 62, 'right_speed': 49},
               {'ground': [968, 692], 'sensor': [968, 692], 'left_speed': 41, 'right_speed': 53},
               {'ground': [968, 702], 'sensor': [968, 702], 'left_speed': 57, 'right_speed': 37},
               {'ground': [968, 712], 'sensor': [968, 712], 'left_speed': 62, 'right_speed': 49},
               {'ground': [970, 738], 'sensor': [970, 738], 'left_speed': 42, 'right_speed': 51},
               {'ground': [970, 738], 'sensor': [970, 738], 'left_speed': 42, 'right_speed': 51},
               {'ground': [970, 751], 'sensor': [970, 751], 'left_speed': 68, 'right_speed': 57},
               {'ground': [970, 746], 'sensor': [970, 746], 'left_speed': 40, 'right_speed': 39},
               {'ground': [970, 744], 'sensor': [970, 744], 'left_speed': 53, 'right_speed': 48},
               {'ground': [972, 728], 'sensor': [972, 728], 'left_speed': 35, 'right_speed': 49},
               {'ground': [971, 729], 'sensor': [971, 729], 'left_speed': 41, 'right_speed': 55},
               {'ground': [971, 729], 'sensor': [971, 729], 'left_speed': 41, 'right_speed': 55},
               {'ground': [970, 727], 'sensor': [970, 727], 'left_speed': 59, 'right_speed': 55},
               {'ground': [971, 727], 'sensor': [971, 727], 'left_speed': 59, 'right_speed': 57},
               {'ground': [970, 738], 'sensor': [970, 738], 'left_speed': 52, 'right_speed': 44},
               {'ground': [970, 724], 'sensor': [970, 724], 'left_speed': 62, 'right_speed': 57},
               {'ground': [970, 708], 'sensor': [970, 708], 'left_speed': 51, 'right_speed': 38},
               {'ground': [970, 719], 'sensor': [970, 719], 'left_speed': 58, 'right_speed': 57},
               {'ground': [970, 702], 'sensor': [970, 702], 'left_speed': 42, 'right_speed': 43},
               {'ground': [970, 715], 'sensor': [970, 715], 'left_speed': 57, 'right_speed': 49},
               {'ground': [970, 705], 'sensor': [970, 705], 'left_speed': 51, 'right_speed': 47},
               {'ground': [970, 713], 'sensor': [970, 713], 'left_speed': 44, 'right_speed': 33},
               {'ground': [970, 695], 'sensor': [970, 695], 'left_speed': 44, 'right_speed': 59},
               {'ground': [970, 695], 'sensor': [970, 695], 'left_speed': 44, 'right_speed': 59},
               {'ground': [970, 707], 'sensor': [970, 707], 'left_speed': 52, 'right_speed': 45},
               {'ground': [970, 707], 'sensor': [970, 707], 'left_speed': 52, 'right_speed': 45},
               {'ground': [970, 718], 'sensor': [970, 718], 'left_speed': 64, 'right_speed': 56},
               {'ground': [970, 705], 'sensor': [970, 705], 'left_speed': 56, 'right_speed': 52},
               {'ground': [970, 705], 'sensor': [970, 705], 'left_speed': 56, 'right_speed': 52},
               {'ground': [970, 695], 'sensor': [970, 695], 'left_speed': 62, 'right_speed': 48},
               {'ground': [970, 702], 'sensor': [970, 702], 'left_speed': 50, 'right_speed': 48},
               {'ground': [969, 712], 'sensor': [969, 712], 'left_speed': 41, 'right_speed': 34},
               {'ground': [970, 708], 'sensor': [970, 708], 'left_speed': 41, 'right_speed': 48},
               {'ground': [970, 703], 'sensor': [970, 703], 'left_speed': 54, 'right_speed': 39},
               {'ground': [970, 703], 'sensor': [970, 703], 'left_speed': 54, 'right_speed': 39},
               {'ground': [970, 703], 'sensor': [970, 703], 'left_speed': 68, 'right_speed': 54},
               {'ground': [971, 698], 'sensor': [971, 698], 'left_speed': 43, 'right_speed': 49},
               {'ground': [970, 682], 'sensor': [970, 682], 'left_speed': 56, 'right_speed': 58},
               {'ground': [969, 679], 'sensor': [969, 679], 'left_speed': 56, 'right_speed': 56},
               {'ground': [968, 685], 'sensor': [968, 685], 'left_speed': 52, 'right_speed': 45},
               {'ground': [969, 702], 'sensor': [969, 702], 'left_speed': 63, 'right_speed': 46},
               {'ground': [970, 712], 'sensor': [970, 712], 'left_speed': 61, 'right_speed': 55},
               {'ground': [970, 738], 'sensor': [970, 738], 'left_speed': 53, 'right_speed': 55},
               {'ground': [970, 732], 'sensor': [970, 732], 'left_speed': 62, 'right_speed': 55},
               {'ground': [970, 731], 'sensor': [970, 731], 'left_speed': 51, 'right_speed': 38}]

l_speed = [x["left_speed"] for x in thymio_data]
r_speed = [x["right_speed"] for x in thymio_data]
avg_speed = [(x["left_speed"] + x["right_speed"]) / 2 for x in thymio_data]
thymio_speed_to_ms = 0.43478260869565216 * 10 ** (-3)
var_speed = np.var([x / thymio_speed_to_ms for x in avg_speed[55:]])
std_speed = np.std([x / thymio_speed_to_ms for x in avg_speed[55:]])
# Variance of speed state and measurements
q_nu = std_speed / 2  # variance on speed state
r_nu = std_speed / 2  # variance on speed measurement

# Variance on position state and measurement
qp = q_nu * SAMPLING_TIME  # 0.04 # variance on position state
rp = r_nu * SAMPLING_TIME  # 0.25 # variance on position measurement
q_theta = 0.1  # variance on angle orientation state
r_theta = 0.1  # variance on angle orientation measurement

# global parameters
R_w = 2.15 * 10 ** (-2)  # Wheel radius
d = 10 * 10 ** (-2)  # distance between wheels
H = np.eye(3)
Q = np.array([[qp, 0, 0], [0, qp, 0], [0, 0, q_theta]])  # Variance of the measurements
R = np.array([[rp, 0, 0], [0, rp, 0], [0, 0, r_theta]])

r1 = 0.0017
r2 = 0.0017
r3 = 0.1

Q = np.array([[r1, 0, 0], [0, r2, 0], [0, 0, r3]])
R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.04]])


def Algorithm_EKF(measurements, mu_km, sig_km, u_k):
    ## Prediction through the a priori estimate
    # estimated mean of the state
    mu_k_pred = np.array([[0.0], [0.0], [0.0]])
    mu_k_pred[0] = mu_km[0, 0] + (u_k[0] + u_k[1]) / 2 * R_w * SAMPLING_TIME * np.sin \
        (mu_km[2, 0] + (u_k[0] - u_k[1]) / d * R_w * SAMPLING_TIME)
    mu_k_pred[1] = mu_km[1, 0] + (u_k[0] + u_k[1]) / 2 * R_w * SAMPLING_TIME * np.cos(
        mu_km[2, 0] + (u_k[0] - u_k[1]) / d * R_w * SAMPLING_TIME)
    mu_k_pred[2] = mu_km[2, 0] + (u_k[0] - u_k[1]) / d * R_w * SAMPLING_TIME

    # Jacobian of the motion model
    G_k = np.eye(3)
    G_k[0, 2] = (u_k[0] + u_k[1]) / 2 * R_w * np.cos(mu_km[2, 0] + 1 / 2 * (u_k[0] - u_k[1]) / d * R_w)
    G_k[1, 2] = -(u_k[0] + u_k[1]) / 2 * R_w * np.sin(mu_km[2, 0] + 1 / 2 * (u_k[0] - u_k[1]) / d * R_w)

    # Estimated covariance of the state
    sig_k_pred = np.dot(G_k, np.dot(sig_km, G_k.T))
    sig_k_pred = sig_k_pred + Q if type(Q) != type(None) else sig_k_pred

    if True:  # camera:
        y = measurements

    else:
        # if no measurements we consider our measurements to be the same as our a priori estimate as to cancel out the effect of innovation
        y = mu_k_pred
        # print('y:',y)
        # print('--------')

    # innovation / measurement residual
    i = y - np.dot(H, mu_k_pred)
    # print('np.dot(H, mu_k_pred): ',np.dot(H, mu_k_pred))
    # print('innovation:', i)
    # print('--------')
    # measurement prediction covariance
    S = np.dot(H, np.dot(sig_k_pred, H.T)) + R

    # Kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = np.dot(sig_k_pred, np.dot(H.T, np.linalg.inv(S)))

    # a posteriori estimate
    # print('mu_k_pred:', mu_k_pred)
    # print('--------')
    x_est = mu_k_pred + np.dot(K, i)
    # print('mu_k_pred + np.dot(K,i): ',mu_k_pred + np.dot(K,i))
    # print('--------')
    sig_est = sig_k_pred - np.dot(K, np.dot(H, sig_k_pred))

    return x_est, sig_est
