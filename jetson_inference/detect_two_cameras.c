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


int main( int argc, char** argv )
{
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
		printf("=======after net->Detect======== \n");
		
		//cam_1
		if( numDetections > 0 )
		{
			LogVerbose("cam_1 : %i objects detected\n", numDetections);
		
			for( int n=0; n < numDetections; n++ )
			{
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
			LogVerbose("cam_2 : %i objects detected\n", numDetections_2);

			for( int n=0 ; n < numDetections_2 ; n++ )
			{
					
				//LogVerbose("\ncam_2 : detected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
				
				LogVerbose("cam_2 : bounding box %i  (%.2f, %.2f)  (%.2f, %.2f)  w=%.2f  h=%.2f\n", n, detections_2[n].Left, detections_2[n].Top, detections_2[n].Right, detections_2[n].Bottom, detections_2[n].Width(), detections_2[n].Height()); 


				LogVerbose("cam_2 : Center coordinate %i : (%.1f, %.1f)\n", n, (detections_2[n].Left + detections_2[n].Right)/2, (detections_2[n].Top + detections_2[n].Bottom)/2);


				cudaDrawCircle(image_2, input_2->GetWidth(), input_2->GetHeight(), (detections_2[n].Left + detections_2[n].Right)/2, (detections_2[n].Top + detections_2[n].Bottom)/2, 5, make_float4(255,0,0,200)); 


				if( detections_2[n].TrackID >= 0 ) // is this a tracked object?
					LogVerbose("cam_2 : tracking  ID %i  status=%i  frames=%i  lost=%i\n", detections_2[n].TrackID, detections_2[n].TrackStatus, detections_2[n].TrackFrames, detections_2[n].TrackLost);

			}
		}


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

