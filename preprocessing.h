#pragma once
/*
 *  preprocessing.h
 *  
 *
 *  Created by  on 18/11/.
 *  Copyright ____. All rights reserved.
 *
 */
#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <ctype.h>
#endif

IplImage preprocessing(IplImage* imgSrc,int new_width, int new_height);
