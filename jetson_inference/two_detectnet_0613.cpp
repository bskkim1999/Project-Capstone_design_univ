/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

//standard of ground height.
//depth(cm)          ground pixel
//35                    292
//36                    294
//37                    296
//38                    298
//39                    300
//40                    302
//41                    304
//42                    306
//etc                   etc



#include "videoSource.h"
#include "videoOutput.h"

#include "detectNet.h"
#include "objectTracker.h"

#include <signal.h>
//modified code
#include "cudaDraw.h"
#include "cudaFont.h"
#include <math.h>
#define PI 3.1415926535


bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		LogVerbose("received SIGINT\n");
		signal_recieved = true;
	}
}

int usage()
{
	printf("usage: detectnet [--help] [--network=NETWORK] [--threshold=THRESHOLD] ...\n");
	printf("                 input [output]\n\n");
	printf("Locate objects in a video/image stream using an object detection DNN.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");
	printf("positional arguments:\n");
	printf("    input           resource URI of input stream  (see videoSource below)\n");
	printf("    output          resource URI of output stream (see videoOutput below)\n\n");

	printf("%s", detectNet::Usage());
	printf("%s", objectTracker::Usage());
	printf("%s", videoSource::Usage());
	printf("%s", videoOutput::Usage());
	printf("%s", Log::Usage());

	return 0;
}

double convert_cm_to_meter(double cm_){
	return cm_ / 100;
}

double convert_pixel_to_meter(double cm_, double pixel_, double object_pixel){

	double pixel_per_cm = pixel_ / cm_;

	return object_pixel / pixel_per_cm / 100;

}


double return_pixel__depth_is_35cm_to_36cm(double object_depth_cm_){

	return (-7)*(object_depth_cm_ - 35) + 227;

}

double return_pixel__depth_is_36cm_to_37cm(double object_depth_cm_){

	return (-7)*(object_depth_cm_ - 36) + 220;
}

double return_pixel__depth_is_37cm_to_38cm(double object_depth_cm_){

	return (-10)*(object_depth_cm_ - 37) + 217;
}

double return_pixel__depth_is_38cm_to_39cm(double object_depth_cm_){

	return (-2)*(object_depth_cm_ - 38) + 207;
}

double return_pixel__depth_is_39cm_to_40cm(double object_depth_cm_){

	return (-5)*(object_depth_cm_ - 39) + 200;
}

double return_pixel__depth_is_40cm_to_41cm(double object_depth_cm_){

	return (-5)*(object_depth_cm_ - 40) + 200;

}

double return_pixel__depth_is_41cm_to_42cm(double object_depth_cm_){

	return (-3)*(object_depth_cm_ - 41) + 196;
}

double return_pixel__depth_is_42cm_to_46cm(double object_depth_cm_){

	return (-4.75)*(object_depth_cm_ - 42) + 192;
}

double return_pixel__depth_is_46cm_to_50cm(double object_depth_cm_){

	return (-3.25)*(object_depth_cm_ - 46) + 173;
}

double return_pixel__depth_is_50cm_to_54cm(double object_depth_cm_){

	return (-2.5)*(object_depth_cm_ - 50) + 160;
}

double return_pixel__depth_is_54cm_to_58cm(double object_depth_cm_){

	return (-2.5)*(object_depth_cm_ - 54) + 150;
}

double return_pixel__depth_is_58cm_to_62cm(double object_depth_cm_){

	return (-2.5)*(object_depth_cm_ - 58) + 140;
}

double return_pixel__depth_is_62cm_to_66cm(double object_depth_cm_){

	return (-1.25)*(object_depth_cm_ - 62) + 130;
}

double return_pixel__depth_is_66cm_to_70cm(double object_depth_cm_){

	return (-1.75)*(object_depth_cm_ - 66) + 125;
}

double median_func(double* array, int len){
	double answer=0;
	double temp=0;
	
	//bubble algorithm
	for(int i=0; i < len ; i++){
		for(int j=0 ; j < (len-1)-i ; j++) {
			if(array[j] > array[j+1]){
				temp = array[j];
				array[j] = array[j+1];
				array[j+1] = temp;
				//printf("!! \n");
			}
			//printf("j : %d \n", j);
		}
		//printf("i : %d \n", i);
	}
	return answer = array[len/2];
}



int main( int argc, char** argv )
{

	int count_object = 0;  //object
	
	//make array object_depth.
	double array_object_depth[10];  //0 1 2 3 4 5 6 7 8 9
	double array_object_width[10];
	double array_object_height[10];
	
	double median_object_width=0;
	double median_object_height=0;
	double median_object_depth=0;


	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);
	printf("===============after cmdLine=============== \n");
	if( cmdLine.GetFlag("help") )
		return usage();


	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		LogError("can't catch SIGINT\n");


	/*
	 * create input stream
	 */
	printf("==============before videoSource::Create for main function================ \n");
	videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));
	//modified code
	videoSource* input_2 = videoSource::Create(cmdLine, ARG_POSITION(1));
	printf("=============after videoSource::Create for main function================= \n");

	if( !input )
	{
		LogError("detectnet(cam_1):  failed to create input stream\n");
		return 1;
	}

	if( !input_2 )
	{
		LogError("detectnet(cam_2)  failed to create input stream \n");
		return 1;
	}

	/*
	 * create output stream
	 */
	printf("============before videoOutput::Create for main function=================== \n");
	//cam_1
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(2));
	//modified code
	//cam_2
	videoOutput* output_2 = videoOutput::Create(cmdLine, ARG_POSITION(3));
	
	printf("===========after videoOutput::Create for main function================= \n");
	if( !output )
	{
		LogError("detectnet(cam_1):  failed to create output stream\n");	
		return 1;
	}
	
	if( !output_2 )
	{
		LogError("detectnet(cam_2):  failed to create output stream\n");
		return 1;
	}



	/*
	 * create detection network
	 */
	printf("===========before detectNet::Create for main function====================== \n");
	//cam_1
	detectNet* net = detectNet::Create(cmdLine);
	//modified code
	//cam_2
	detectNet* net_2 = detectNet::Create(cmdLine);
	printf("==========after detectNet::Create for main function====================== \n");
	if( !net )
	{
		LogError("detectnet:  failed to load detectNet model\n");
		return 1;
	}

	if( !net_2 )
	{
		LogError("detectnet_2:  failed to load detectNet model\n");
		return 1;
	}

	// parse overlay flags
	printf("==========before detectNet::OverlayFlagsFromStr for main function ======= \n");
	const uint32_t overlayFlags = detectNet::OverlayFlagsFromStr(cmdLine.GetString("overlay", "box,labels,conf"));
	//modified code
	const uint32_t overlayFlags_2 = detectNet::OverlayFlagsFromStr(cmdLine.GetString("overlay", "box,labels,conf"));
	printf("=========after detectNet::OverlayFlagsFromStr for main function ======== \n");

	/*
	 * processing loop
	 */
	while( !signal_recieved )
	{
		// capture next image
		printf("========start while loop========== \n");
		//cam_1
		uchar3* image = NULL;
		int status = 0;
		//cam_2
		//modified code
		uchar3* image_2 = NULL;
		int status_2 = 0;

		double cam_1_x_center = 0;
		double cam_1_y_center = 0;
		double cam_1_height = 0;

		double cam_2_x_center = 0;
		double cam_2_y_center = 0;
		double cam_2_height = 0;
		
		bool cam_1_object_detected = false;
		bool cam_2_object_detected = false;

		//double cam_1_confidence = 0;
		//double cam_2_confidence = 0;

		printf("=======start input->Capture====== \n");
		//cam_1
		if( !input->Capture(&image, &status) )
		{
			if( status == videoSource::TIMEOUT )
				continue;
			
			break; // EOS
		}
		//cam_2
		//modified code
		if( !input_2->Capture(&image_2, &status_2) )
		{
			if( status_2 == videoSource::TIMEOUT )
				continue;

			break;
		}

		printf("========after input->Capture====== \n");
		// detect objects in the frame
		detectNet::Detection* detections = NULL;
		//modified code
		detectNet::Detection* detections_2 = NULL;
		
		printf("========before net->Detect======= \n");
		//cam_1
		const int numDetections = net->Detect(image, input->GetWidth(), input->GetHeight(), &detections, overlayFlags);
		//modified code
		//cam_2
		const int numDetections_2 = net_2->Detect(image_2, input_2->GetWidth(), input_2->GetHeight(), &detections_2, overlayFlags_2);

		//modified code : to make letters on the monitor
		char buffer_[500];
		char buffer_2[500];
		float4 color_ = make_float4(255, 255, 255, 255); //letter's color : cam_1 and cam_2
		cudaFont* font_center_coordinate = NULL; //cam_1
		cudaFont* font_center_coordinate_2 = NULL; //cam_2
		font_center_coordinate = cudaFont::Create(adaptFontSize(input->GetWidth())); //cam_1
		font_center_coordinate_2 = cudaFont::Create(adaptFontSize(input->GetWidth())); //cam_2
		///////////////////////////////////////////////////

		printf("=======after net->Detect======== \n");
		

		//cam_1
		if( numDetections > 0 )
		{
			//modified code//////////
			cam_1_object_detected = true;
			LogVerbose("cam_1 : %i objects detected\n", numDetections);
		
			for( int n=0; n < numDetections; n++ )
			{
				
				//modified code
				const int2 position_ = make_int2((detections[n].Left + detections[n].Right) / 2, (detections[n].Top + detections[n].Bottom) / 2 - 50);
				sprintf(buffer_, "(%.1f, %.1f)", (detections[n].Left + detections[n].Right) / 2, (detections[n].Top + detections[n].Bottom)/2 );
				font_center_coordinate->OverlayText(image, IMAGE_RGB8, input->GetWidth(), input->GetHeight(), buffer_, position_.x, position_.y, color_);

				cam_1_x_center = (detections[n].Left + detections[n].Right) / 2;
				cam_1_y_center = (detections[n].Top + detections[n].Bottom) / 2;
				//cam_1_confidence = detections[n].Confidence;
				//cam_1_height = detections[n].Height();
				
				
				//LogVerbose("\ncam_1 : detected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
				LogVerbose("cam_1 : bounding box %i  (%.2f, %.2f)  (%.2f, %.2f)  w=%.2f  h=%.2f\n", n, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height()); 
				//modified code
				LogVerbose("cam_1 : Center coordinate %i : (%.1f, %.1f)\n", n, (detections[n].Left + detections[n].Right)/2, (detections[n].Top + detections[n].Bottom)/2);
				cudaDrawCircle(image, input->GetWidth(), input->GetHeight(), (detections[n].Left + detections[n].Right)/2, (detections[n].Top + detections[n].Bottom)/2, 5, make_float4(255,0,0,200)); 

				if( detections[n].TrackID >= 0 ) // is this a tracked object?
					LogVerbose("cam_1 : tracking  ID %i  status=%i  frames=%i  lost=%i\n", detections[n].TrackID, detections[n].TrackStatus, detections[n].TrackFrames, detections[n].TrackLost);
			}
		}
		//modified code
		//cam_2
		if( numDetections_2 > 0 )
		{

			//modified code/////
			cam_2_object_detected = true;
			LogVerbose("cam_2 : %i objects detected\n", numDetections_2);

			for( int n=0 ; n < numDetections_2 ; n++ )
			{
				//modified code//////////////
				const int2 position_2 = make_int2((detections_2[n].Left + detections_2[n].Right) / 2, (detections_2[n].Top + detections_2[n].Bottom) / 2 - 50);
				sprintf(buffer_2, "(%.1f, %.1f)", (detections_2[n].Left + detections_2[n].Right) / 2, (detections_2[n].Top + detections_2[n].Bottom) / 2);
				font_center_coordinate->OverlayText(image_2, IMAGE_RGB8, input_2->GetWidth(), input_2->GetHeight(), buffer_2, position_2.x, position_2.y, color_);

				cam_2_x_center = (detections_2[n].Left + detections_2[n].Right) / 2;
				cam_2_y_center = (detections_2[n].Top + detections_2[n].Bottom) / 2;
				//cam_2_confidence = detections_2[n].Confidence;
				//cam_2_height = detections_2[n].Height();

				//LogVerbose("\ncam_2 : detected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
				
				LogVerbose("cam_2 : bounding box %i  (%.2f, %.2f)  (%.2f, %.2f)  w=%.2f  h=%.2f\n", n, detections_2[n].Left, detections_2[n].Top, detections_2[n].Right, detections_2[n].Bottom, detections_2[n].Width(), detections_2[n].Height()); 
				LogVerbose("cam_2 : Center coordinate %i : (%.1f, %.1f)\n", n, (detections_2[n].Left + detections_2[n].Right)/2, (detections_2[n].Top + detections_2[n].Bottom)/2);


				cudaDrawCircle(image_2, input_2->GetWidth(), input_2->GetHeight(), (detections_2[n].Left + detections_2[n].Right)/2, (detections_2[n].Top + detections_2[n].Bottom)/2, 5, make_float4(255,0,0,200)); 


				if( detections_2[n].TrackID >= 0 ) // is this a tracked object?
					LogVerbose("cam_2 : tracking  ID %i  status=%i  frames=%i  lost=%i\n", detections_2[n].TrackID, detections_2[n].TrackStatus, detections_2[n].TrackFrames, detections_2[n].TrackLost);

			}
		}
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//modified code : calculate depth.
		double cam_distance = 22.5;  //cm
		double webcam_dFov = (11 * PI) / 36;  //55 degree -> radian
		double angle1, angle2, angle3, robot_height_pixel;
		double object_depth;
		double object_height;
		double object_width;
		double robot_height_cm = 22.5;  //cm
		double cam_width = 1280.0;  //width is 1280 pixels.
		double object_y_average;
		double cam_height_average_pixel;

		//width variable.
		double beta_1;
		double O1;
		double beta_2;
		double O2;
		double cam_mid_location = cam_distance / 2.0;

		//cam_1 and cam_2. angle1, angle2, angle3 are radian.
		angle1 = (cam_width - cam_1_x_center) * (webcam_dFov / cam_width) + (PI - webcam_dFov) / 2;
		angle2 = cam_2_x_center * (webcam_dFov / cam_width) + (PI - webcam_dFov) / 2;
		angle3 = PI - angle1 - angle2;

		if (cam_1_object_detected == true && cam_2_object_detected == true  ) {
			//calculate depth.
			object_depth = (cam_distance * sin(angle1) * sin(angle2)) / sin(angle3) ;

			//calculate width.
			//if object is on left side.
				if( cam_2_x_center < (cam_width - cam_1_x_center) ){
					//object_width = convert_pixel_to_meter( 5.0, tmp, ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - cam_2_x_center));
					object_width = (object_depth - (cam_mid_location * tan(angle2)) ) / tan(angle2);
				}

				//if object is on right side.
				else{
					if(cam_2_x_center > (cam_width - cam_1_x_center)){
						//object_width = convert_pixel_to_meter(5.0, tmp, (-1) * ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - (cam_width - cam_1_x_center)));  
						object_width = (object_depth - (cam_mid_location * tan(angle1)) ) / tan(angle1);
					}
				//object is in the middle.
					else{
						object_width = 0;
					}

				}
			
			/////////////////////////////////////////////////////////////////////////////////////////////////

			//calculate height : algorithm splits according to depth.
			if(object_depth >=35 && object_depth < 36){
				double tmp = return_pixel__depth_is_35cm_to_36cm(object_depth);
				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average - 293 );
			}

			else if(object_depth >= 36 && object_depth < 37){
				
				double tmp = return_pixel__depth_is_36cm_to_37cm(object_depth);
				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average - 295 );
				

			}
			
			else if(object_depth >= 37 && object_depth < 38){
				
				double tmp = return_pixel__depth_is_37cm_to_38cm(object_depth);
				

				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average - 297 );
			}

			else if(object_depth >= 38 && object_depth < 39){
				
				double tmp = return_pixel__depth_is_38cm_to_39cm(object_depth);
				
				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average - 299 );
			}

			else if(object_depth >= 39 && object_depth < 40){
				double tmp = return_pixel__depth_is_39cm_to_40cm(object_depth);
				

				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average - 301);
			}


			else if(object_depth >= 40 && object_depth < 41){
				
				double tmp = return_pixel__depth_is_40cm_to_41cm(object_depth);
				
				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average - 303 );
			}

			else if(object_depth >= 41 && object_depth < 42){
				
				double tmp = return_pixel__depth_is_41cm_to_42cm(object_depth);
				
				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average - 305 );
			}

			else if(object_depth >= 42 && object_depth < 46){
				
				double tmp = return_pixel__depth_is_42cm_to_46cm(object_depth);
				

				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average - 310 );
			}

			else if(object_depth >= 46 && object_depth < 50){
				
				double tmp = return_pixel__depth_is_46cm_to_50cm(object_depth);
				

				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average - 318 );
			}

			else if(object_depth >= 50 && object_depth < 54){
				
				double tmp = return_pixel__depth_is_50cm_to_54cm(object_depth);
				
				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average - 326 );
			}
			
			else if(object_depth >= 54 && object_depth < 58){
				
				double tmp = return_pixel__depth_is_54cm_to_58cm(object_depth);
				
				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average - 334 );
			}

			else if(object_depth >= 58 && object_depth < 62){
				
				double tmp = return_pixel__depth_is_58cm_to_62cm(object_depth);
				

				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average - 342 );
			}

			else if(object_depth >= 62 && object_depth < 66){
				
				double tmp = return_pixel__depth_is_62cm_to_66cm(object_depth);
				
				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average - 350 );
			}

			else if(object_depth >= 66 && object_depth < 70){
				
				double tmp = return_pixel__depth_is_66cm_to_70cm(object_depth);

				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average - 358 );
			}

			//printf("object_depth : %.6f \n", object_depth);
			//printf("object_width : %.6f \n", object_width);
			//printf("object_height : %.6f \n", object_height);

			//count object. ////////////////////////////////////////////////
			//count_object  ==> 0 1 2 3 4 5 6 7 8 9
			if(count_object < 10){
				
				array_object_depth[count_object] = object_depth;
				array_object_width[count_object] = object_width;
				array_object_height[count_object] = object_height;

				count_object++;
			}

			else{
						
				count_object = 10;   //to fix count_object.
				printf("hello \n");
				//calculate median.
				median_object_depth = convert_cm_to_meter( median_func(array_object_depth, 10) );    //cm -> meter
				median_object_width = convert_cm_to_meter( median_func(array_object_width, 10) );    //cm -> meter
				median_object_height = median_func(array_object_height, 10);  //meter
				
				
				printf("median_object_depth : %.6f \n", median_object_depth); 
				printf("median_object_width : %.6f \n", median_object_width); 
				printf("median_object_height : %.6f \n", median_object_height); 

				break;
				
			}
			//////////////////
			printf("count_object : %d \n", count_object);

		}
		
		//Not detected.
		else {
			printf("object_depth : not calculated \n");
			printf("object_height : not calculated \n");
			printf("object_width : not calculated \n");
		}



		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// render outputs
		// cam_1
		if( output != NULL )
		{
			//printf("=========start render outputs (if)=============== \n");
			output->Render(image, input->GetWidth(), input->GetHeight());
			
			
			//printf("========check render outputs_1 ================= \n");
			// update the status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, precisionTypeToStr(net->GetPrecision()), net->GetNetworkFPS());
			output->SetStatus(str);
			//printf("=========check render outputs_2 =============== \n");
			// check if the user quit
			if( !output->IsStreaming() )
				break;
		}
		
		// print out timing info
		//net->PrintProfilerTimes();


		//modified code
		//cam_2
		if( output_2 != NULL )
		{
			output_2->Render(image_2, input_2->GetWidth(), input_2->GetHeight());
			char str_2[256];

			sprintf(str_2, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, precisionTypeToStr(net_2->GetPrecision()), net_2->GetNetworkFPS());
			output_2->SetStatus(str_2);

			if( !output_2->IsStreaming() )
				break;
			
		}


		printf("===============================while loop (main) ==========================\n");

	}//while loop
	

	/*
	 * destroy resources
	 */
	LogVerbose("detectnet:  shutting down...\n");
	
	SAFE_DELETE(input);
	SAFE_DELETE(output);
	SAFE_DELETE(net);

	LogVerbose("detectnet:  shutdown complete.\n");
	return 0;
}

