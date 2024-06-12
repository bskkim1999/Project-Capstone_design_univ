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

#include "videoSource.h"
#include "videoOutput.h"

#include "detectNet.h"
#include "objectTracker.h"

#include <signal.h>
//modified code
#include "cudaDraw.h"
#include "cudaFont.h"
#include "uart.h"
#include <math.h>
#define PI 3.1415926535
#include <unistd.h>

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


bool check_if_destination(Uart *u_){
	if( strcmp(u_->readUart() , "o#") == 0){
		return true;
	}

	else if( strcmp(u_->readUart() ,"x#") == 0){
		return false;
	}

	else{
		return false;
	}
}

double convert_pixel_to_meter(double cm_, double pixel_, double object_pixel){

	double pixel_per_cm = pixel_ / cm_;

	return object_pixel / pixel_per_cm / 100;

}


double return_pixel__depth_is_35cm_to_36cm(double object_depth_cm_){

	return -7*(object_depth_cm_ - 35) + 227;

}

double return_pixel__depth_is_36cm_to_37cm(double object_depth_cm_){

	return -7*(object_depth_cm_ - 36) + 220;
}

double return_pixel__depth_is_37cm_to_38cm(double object_depth_cm_){

	return -10*(object_depth_cm_ - 37) + 217;
}

double return_pixel__depth_is_38cm_to_39cm(double object_depth_cm_){

	return -2*(object_depth_cm_ - 38) + 207;
}

double return_pixel__depth_is_39cm_to_40cm(double object_depth_cm_){

	return -5*(object_depth_cm_ - 39) + 200;
}

double return_pixel__depth_is_40cm_to_41cm(double object_depth_cm_){

	return -5*(object_depth_cm_ - 40) + 200;

}

double return_pixel__depth_is_41cm_to_42cm(double object_depth_cm_){

	return -3*(object_depth_cm_ - 41) + 196;
}

int median_func(double* array, int len){
	int answer=0;
	int i,j,temp=0;
	
	//bubble algorithm
	for(i=0; i < len ; i++){
		for(j=0 ; j = (len-1)-i ; j++) {
			if(array[j] > array[j+1]){
				temp = array[j];
				array[j] = array[j+1];
				array[j+1] = temp;
			}
		}
	}
	return answer = array[len/2];
}


int main( int argc, char** argv )
{
	
	Uart u;  //uart serial communication.
	int count_object = 0;  //object
	
	char msg_0[50];
	char msg_1[50];
	char msg_2[50];
	char msg_3[50];

	char number_zero = '0';
	char letter_a = 'a';
	double data_zero = 0;
	
	int only_one_1 = 1;
	int only_one_2 = 1;
	
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

	
	//modified code : check if destination ?
	while( check_if_destination(&u) == false ) {
		printf("===============Loading if robot is arrived at destination.===========\n");
	}
	
	//Protocol : 0a(0)a(0)a(0)
	sprintf(msg_0, "%c%c%.6f%c%.6f%c%.6f", number_zero, letter_a, data_zero, letter_a, data_zero, letter_a, data_zero);  
	u.sendUart(msg_0);    //send.


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
		
		double cam_1_x_center=0;
	        double cam_1_y_center=0;
	        double cam_2_x_center=0;
	        double cam_2_y_center=0;

		bool cam_1_object_detected = false;
		bool cam_2_object_detected = false;
		

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
		detectNet::Detection* detections = NULL;  //detections is struct pointer variable.
		//modified code
		detectNet::Detection* detections_2 = NULL; //detections_2 is struct pointer variable.
		
		printf("========before net->Detect======= \n");
		//cam_1
		const int numDetections = net->Detect(image, input->GetWidth(), input->GetHeight(), &detections, overlayFlags);
		//modified code
		//cam_2
		const int numDetections_2 = net_2->Detect(image_2, input_2->GetWidth(), input_2->GetHeight(), &detections_2, overlayFlags_2);



		//modified code : to make letters on the monitor
		char buffer_[500];  //cam_1
		char buffer_2[500]; //cam_2
		float4 color_ = make_float4(255,255,255,255); //letter's color : cam_1 and cam_2
		cudaFont* font_center_coordinate = NULL; //cam_1
		cudaFont* font_center_coordinate_2 = NULL; //cam_2
		font_center_coordinate = cudaFont::Create( adaptFontSize( input->GetWidth() ) ); //cam_1
		font_center_coordinate_2 = cudaFont::Create( adaptFontSize( input->GetWidth() ) ); //cam_2


		printf("=======after net->Detect======== \n");
		
		//cam_1
		if( numDetections > 0 )
		{
			//modified code
			cam_1_object_detected = true;

			LogVerbose("cam_1 : %i objects detected\n", numDetections);
		
			for( int n=0; n < numDetections; n++ )
			{
				
				//modified code
				const int2 position_ = make_int2( (detections[n].Left+detections[n].Right)/2, (detections[n].Top+detections[n].Bottom)/2 - 50 );

				sprintf(buffer_, "(%.1f, %.1f)", (detections[n].Left+detections[n].Right)/2, (detections[n].Top+detections[n].Bottom)/2 );

				font_center_coordinate->OverlayText( image, IMAGE_RGB8, input->GetWidth(), input->GetHeight(), buffer_, position_.x, position_.y, color_ );
				
				
				cam_1_x_center = (detections[n].Left + detections[n].Right)/2;
				cam_1_y_center = (detections[n].Top + detections[n].Bottom)/2;


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
			//modified code
			cam_2_object_detected = true;
			LogVerbose("cam_2 : %i objects detected\n", numDetections_2);

			for( int n=0 ; n < numDetections_2 ; n++ )
			{
					
				//modified code
				const int2 position_2 = make_int2( (detections_2[n].Left+detections_2[n].Right)/2, (detections_2[n].Top+detections_2[n].Bottom)/2 - 50);

				sprintf(buffer_2, "(%.1f, %.1f)", (detections_2[n].Left+detections_2[n].Right)/2, (detections_2[n].Top+detections_2[n].Bottom)/2 );

				font_center_coordinate_2->OverlayText( image_2, IMAGE_RGB8, input_2->GetWidth(), input_2->GetHeight(), buffer_2, position_2.x, position_2.y, color_);
				
				
				
				cam_2_x_center = (detections_2[n].Left + detections_2[n].Right)/2;
				cam_2_y_center = (detections_2[n].Bottom + detections_2[n].Top)/2;



				//LogVerbose("\ncam_2 : detected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
				
				LogVerbose("cam_2 : bounding box %i  (%.2f, %.2f)  (%.2f, %.2f)  w=%.2f  h=%.2f\n", n, detections_2[n].Left, detections_2[n].Top, detections_2[n].Right, detections_2[n].Bottom, detections_2[n].Width(), detections_2[n].Height()); 


				LogVerbose("cam_2 : Center coordinate %i : (%.1f, %.1f)\n", n, (detections_2[n].Left + detections_2[n].Right)/2, (detections_2[n].Top + detections_2[n].Bottom)/2);


				cudaDrawCircle(image_2, input_2->GetWidth(), input_2->GetHeight(), (detections_2[n].Left + detections_2[n].Right)/2, (detections_2[n].Top + detections_2[n].Bottom)/2, 5, make_float4(255,0,0,200)); 


				if( detections_2[n].TrackID >= 0 ) // is this a tracked object?
					LogVerbose("cam_2 : tracking  ID %i  status=%i  frames=%i  lost=%i\n", detections_2[n].TrackID, detections_2[n].TrackStatus, detections_2[n].TrackFrames, detections_2[n].TrackLost);

			}
		}
		////////////////////////////////////////////////////////////////////////////////////////////
	

		//modified code : calculate object depth, width, height.
	
		double cam_distance = 22.5;   // cm
		double webcam_dFov = (11*PI)/36;  //55 degree -> radian
		double angle1, angle2, angle3;
		double cam_width = 1280.0;  //width is 1280 pixels.
		double robot_height_pixel;
		
		double object_width;
		double object_height;
		double object_depth;
		double object_y_average;
		
		

		char separated_letter = 'a';
		char number_one = '1';
		char number_two = '2';
		char number_three = '3';

		//make array object_depth.
		double array_object_depth[10];  //0 1 2 3 4 5 6 7 8 9
		double array_object_width[10];
		double array_object_height[10];


		//cam_1 and cam_2
		angle1 = (cam_width - cam_1_x_center) * (webcam_dFov / cam_width) + (PI - webcam_dFov)/2;
		angle2 = cam_2_x_center * (webcam_dFov / cam_width) + (PI - webcam_dFov)/2;
		angle3 = PI - angle1 - angle2;
		
		//Detected.
		if(cam_1_object_detected == true && cam_2_object_detected == true){
			

			////////////////////////////////////////////////////////////////////////////////////////
			//calculate depth.
			object_depth = ( cam_distance * sin(angle1) * sin(angle2) ) / sin(angle3);
			
			//////////////////////////////////////////////////////////////////////////
			
			//algorithm splits according to depth.
			if(object_depth >=35 && object_depth < 36){
				//calculate width./////////////
				double tmp = return_pixel__depth_is_35cm_to_36cm(object_depth);
				//if object is on left side.
				if( cam_2_x_center < (cam_width - cam_1_x_center) ){
					object_width = convert_pixel_to_meter( 5.0, tmp, ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - cam_2_x_center));
				}

				//if object is on right side.
				else{
					if(cam_2_x_center > (cam_width - cam_1_x_center)){
						object_width = convert_pixel_to_meter(5.0, tmp, (-1) * ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - (cam_width - cam_1_x_center)));  
					}
				//object is in the middle.
					else{
						object_width = 0;
					}

				}

				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average );
			}

			else if(object_depth >= 36 && object_depth < 37){
				//calculate width./////////////
				double tmp = return_pixel__depth_is_36cm_to_37cm(object_depth);
				//if object is on left side.
				if( cam_2_x_center < (cam_width - cam_1_x_center) ){
					object_width = convert_pixel_to_meter( 5.0, tmp, ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - cam_2_x_center));
				}

				//if object is on right side.
				else{
					if(cam_2_x_center > (cam_width - cam_1_x_center)){
						object_width = convert_pixel_to_meter(5.0, tmp, (-1) * ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - (cam_width - cam_1_x_center)));  
					}
				//object is in the middle.
					else{
						object_width = 0;
					}

				}

				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average );
				

			}
			
			else if(object_depth >= 37 && object_depth < 38){
				//calculate width./////////////
				double tmp = return_pixel__depth_is_37cm_to_38cm(object_depth);
				//if object is on left side.
				if( cam_2_x_center < (cam_width - cam_1_x_center) ){
					object_width = convert_pixel_to_meter( 5.0, tmp, ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - cam_2_x_center));
				}

				//if object is on right side.
				else{
					if(cam_2_x_center > (cam_width - cam_1_x_center)){
						object_width = convert_pixel_to_meter(5.0, tmp, (-1) * ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - (cam_width - cam_1_x_center)));  
					}
				//object is in the middle.
					else{
						object_width = 0;
					}

				}

				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average );
			}

			else if(object_depth >= 38 && object_depth < 39){
				//calculate width./////////////
				double tmp = return_pixel__depth_is_38cm_to_39cm(object_depth);
				//if object is on left side.
				if( cam_2_x_center < (cam_width - cam_1_x_center) ){
					object_width = convert_pixel_to_meter( 5.0, tmp, ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - cam_2_x_center));
				}

				//if object is on right side.
				else{
					if(cam_2_x_center > (cam_width - cam_1_x_center)){
						object_width = convert_pixel_to_meter(5.0, tmp, (-1) * ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - (cam_width - cam_1_x_center)));  
					}
				//object is in the middle.
					else{
						object_width = 0;
					}

				}

				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average );
			}

			else if(object_depth >= 39 && object_depth < 40){
				//calculate width./////////////
				double tmp = return_pixel__depth_is_39cm_to_40cm(object_depth);
				//if object is on left side.
				if( cam_2_x_center < (cam_width - cam_1_x_center) ){
					object_width = convert_pixel_to_meter( 5.0, tmp, ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - cam_2_x_center));
				}

				//if object is on right side.
				else{
					if(cam_2_x_center > (cam_width - cam_1_x_center)){
						object_width = convert_pixel_to_meter(5.0, tmp, (-1) * ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - (cam_width - cam_1_x_center)));  
					}
				//object is in the middle.
					else{
						object_width = 0;
					}

				}

				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average );
			}


			else if(object_depth >= 40 && object_depth < 41){
				//calculate width./////////////
				double tmp = return_pixel__depth_is_40cm_to_41cm(object_depth);
				//if object is on left side.
				if( cam_2_x_center < (cam_width - cam_1_x_center) ){
					object_width = convert_pixel_to_meter( 5.0, tmp, ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - cam_2_x_center));
				}

				//if object is on right side.
				else{
					if(cam_2_x_center > (cam_width - cam_1_x_center)){
						object_width = convert_pixel_to_meter(5.0, tmp, (-1) * ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - (cam_width - cam_1_x_center)));  
					}
				//object is in the middle.
					else{
						object_width = 0;
					}

				}

				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average );
			}

			else if(object_depth >= 41 && object_depth < 42){
				//calculate width./////////////
				double tmp = return_pixel__depth_is_41cm_to_42cm(object_depth);
				//if object is on left side.
				if( cam_2_x_center < (cam_width - cam_1_x_center) ){
					object_width = convert_pixel_to_meter( 5.0, tmp, ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - cam_2_x_center));
				}

				//if object is on right side.
				else{
					if(cam_2_x_center > (cam_width - cam_1_x_center)){
						object_width = convert_pixel_to_meter(5.0, tmp, (-1) * ((cam_2_x_center + (cam_width - cam_1_x_center))/2 - (cam_width - cam_1_x_center)));  
					}
				//object is in the middle.
					else{
						object_width = 0;
					}

				}

				//calculate height.////////////
				object_y_average = (cam_1_y_center + cam_2_y_center) / 2;
				object_height = convert_pixel_to_meter( 5.0, tmp, 720-object_y_average );
			}
			
			//count object. ////////////////////////////////////////////////
			//count_object  ==> 0 1 2 3 4 5 6 7 8 9
			if(count_object < 10){
				count_object++;
				array_object_depth[count_object] = object_depth;
				array_object_width[count_object] = object_width;
				array_object_height[count_object] = object_height;
			}

			else{
						
				count_object = 10;   //to fix count_object.
				
				//calculate median.
				median_object_depth = median_func(array_object_depth, 10);
				median_object_width = median_func(array_object_width, 10);
				median_object_height = median_func(array_object_height, 10);

				if(only_one_2 == 1){
					//Protocol : 2a(0)a(0)a(0)
					sprintf(msg_2, "%c%c%.6f%c%.6f%c%.6f", number_two, separated_letter, data_zero, separated_letter, data_zero, separated_letter, data_zero);
					u.sendUart(msg_2);  //send.
					only_one_2--;
				}

				//Protocol : 3a(width)a(height)a(depth)
				sprintf(msg_3, "%c%c%.6f%c%.6f%c%.6f", number_three, separated_letter, median_object_width, separated_letter, median_object_height, separated_letter, median_object_depth);
				u.sendUart(msg_3);  //send.

				while(1){
					printf("===================End===================== \n");
				}
				
			}
			///////////////////////////////////////////////////////////////////////////////////

			
		}

		//Not detected.
		else{
			
			printf("object_depth : not calculated \n");
			printf("object_width : not calculated \n");
			printf("object_height : not calculated \n");
			
			if(only_one_1 == 1){
				//Protocol : 1a(0)a(0)a(0)
				sprintf(msg_1, "%c%c%.6f%c%.6f%c%.6f", number_one, letter_a, data_zero, letter_a, data_zero, letter_a, data_zero);
				u.sendUart(msg_1);  //send.
				only_one_1 --;
			}
			
			//initialize.
			for(int i=0 ; i<10 ; i++){
				array_object_depth[i] = 0;
				array_object_width[i] = 0;
				array_object_height[i] = 0;
			}
			count_object = 0;   //initialize.

		}
		//////////////////////////////////////////////////////////////////////////////////////////////////
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
		//sleep(10);
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

