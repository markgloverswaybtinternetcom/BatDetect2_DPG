//
//  main.c
//  sndfile_tutorial
//
//  Created by Matthew Hosack on 11/28/14.
//  Copyright (c) 2014 Matthew Hosack. All rights reserved.
//

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "sndfile.h"
#include "portaudio.h"

#define FRAMES_PER_BUFFER   (512)

typedef struct
{
    SNDFILE     *file;
    SF_INFO      info;
	sf_count_t   length;
    sf_count_t   position;
	float        gain;
} callback_data_s;

static int callback(    const void *input, void *output,
                        unsigned long frameCount,
                        const PaStreamCallbackTimeInfo* timeInfo,
                        PaStreamCallbackFlags statusFlags,
                        void *userData );

int main(int argc, const char * argv[])
{
    SNDFILE *file;
    PaStream *stream;
    PaError error;
    callback_data_s data;

    /* Check cli arguments */
    if(argc < 2)
    {
        fprintf(stderr, "wavplay <filename> <start_sample> <end_sample> <sample_rate>\n");
        return 1;
    }
    
    /* Open the soundfile */
    data.file = sf_open(argv[1], SFM_READ, &data.info);
    if (sf_error(data.file) != SF_ERR_NO_ERROR)
    {
        fprintf(stderr, "wavplay %s\n", sf_strerror(data.file));
        fprintf(stderr, "File: %s\n", argv[1]);
        return 1;
    }
    fprintf(stderr, "wavplay sf_open samplerate=%d, channels=%d, format=%d, sections=%d, seekable=%d\n", 
        data.info.samplerate, data.info.channels, data.info.format, data.info.sections, data.info.seekable);
    
    /* init portaudio */
    error = Pa_Initialize();
    if(error != paNoError)
    {
        fprintf(stderr, "wavplay Problem initializing\n");
        return 1;
    }
	long sample_rate = data.info.samplerate;
	data.length = data.info.frames;
	data.position = 0;
	/* If start and stop samples are provided, set sf_seek to start sample */
    if (argc == 6)
    {
        sample_rate = strtol(argv[4], NULL, 10);
        long start_sample = strtol(argv[2], NULL, 10);
        sf_count_t frames = sf_seek(data.file, start_sample * data.info.channels, SEEK_SET);
        fprintf(stderr, "wavplay sf_seek frames=%ld\n", frames); //On success sf_seek returns the current position in (multi-channel) samples from the start of the file
        long stop_sample = strtol(argv[3], NULL, 10);
        data.length = stop_sample - start_sample;
        data.position = 0;
        sample_rate = strtol(argv[4], NULL, 10);
        data.gain = atof(argv[5]);
        fprintf(stderr, "wavplay %s start_sample=%ld, stop_sample=%ld, sample_rate=%ld, gain=%f\n", argv[1], start_sample, stop_sample, sample_rate, data.gain);
    }

    /* Open PaStream with values read from the file */
    error = Pa_OpenDefaultStream(&stream
                                 ,0                     /* no input */
                                 ,data.info.channels         /* stereo out */
                                 ,paFloat32             /* floating point */
                                 ,sample_rate
                                 , FRAMES_PER_BUFFER
                                 ,callback
                                 ,&data);        /* our sndfile data struct */

    if(error != paNoError)
    {
        fprintf(stderr, "wavplay Problem opening Default Stream\n");
        return 1;
    }
    
    /* Start the stream */
    error = Pa_StartStream(stream);
    if(error != paNoError)
    {
        fprintf(stderr, "wavplay Problem opening starting Stream\n");
        return 1;
    }

    /* Run until EOF is reached */
    while(Pa_IsStreamActive(stream))
    {
        Pa_Sleep(100);
    }

    /* Close the soundfile */
    sf_close(data.file);

    /*  Shut down portaudio */
    error = Pa_CloseStream(stream);
    if(error != paNoError)
    {
        fprintf(stderr, "wavplay Problem closing stream\n");
        return 1;
    }
    
    error = Pa_Terminate();
    if(error != paNoError)
    {
        fprintf(stderr, "wavplay Problem terminating\n");
        return 1;
    }
    
    return 0;
}

static int callback
    (const void                     *input
    ,void                           *output
    ,unsigned long                   frameCount
    ,const PaStreamCallbackTimeInfo *timeInfo
    ,PaStreamCallbackFlags           statusFlags
    ,void                           *userData
    )
{
    float           *out;
    callback_data_s *p_data = (callback_data_s*)userData;
    sf_count_t       num_read;

    out = (float*)output;
    p_data = (callback_data_s*)userData;
    sf_count_t left = p_data->length - p_data->position;
	if (left < frameCount) frameCount = left;

    /* clear output buffer */
    memset(out, 0, sizeof(float) * frameCount * p_data->info.channels);

    /* read directly into output buffer */
    num_read = sf_read_float(p_data->file, out, frameCount * p_data->info.channels);
    for (sf_count_t i = 0; i < num_read; i++)
    {
        out[i] *= p_data->gain;
	}   
    p_data->position += frameCount;
 
    /*  If we couldn't read a full frameCount of samples we've reached EOF */
    if (num_read < FRAMES_PER_BUFFER * p_data->info.channels)
    {
        return paComplete;
    }
    
	return paContinue;  // only one frame is needed
}
