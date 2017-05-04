/*
** Copyright (C) 2008-2016 Erik de Castro Lopo <erikd@mega-nerd.com>
**
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**
**     * Redistributions of source code must retain the above copyright
**       notice, this list of conditions and the following disclaimer.
**     * Redistributions in binary form must reproduce the above copyright
**       notice, this list of conditions and the following disclaimer in
**       the documentation and/or other materials provided with the
**       distribution.
**     * Neither the author nor the names of any contributors may be used
**       to endorse or promote products derived from this software without
**       specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
** TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
** PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
** CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
** EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
** PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
** OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
** WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
** OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
** ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <ctype.h>
#include <float.h>
#include <fftw3.h>
#include <math.h>
#include <time.h>

#include <sndfile.h>

#define	BLOCK_SIZE 4096
//#define	BLOCK_SIZE 16 

#ifdef DBL_DECIMAL_DIG
	#define OP_DBL_Digs (DBL_DECIMAL_DIG)
#else
	#ifdef DECIMAL_DIG
		#define OP_DBL_Digs (DECIMAL_DIG)
	#else
		#define OP_DBL_Digs (DBL_DIG + 3)
	#endif
#endif

static void
print_usage (char *progname)
{	printf ("\nUsage : %s [--full-precision] <input file> <output file>\n", progname) ;
	puts ("\n"
		"    Where the output file will contain a line for each frame\n"
		"    and a column for each channel.\n"
		) ;

} /* print_usage */

static void
convert_to_text (SNDFILE * infile, SNDFILE * impulsefile, SNDFILE * finalfile, FILE * outfile, int channels, int full_precision)
{	
    float buf[BLOCK_SIZE], bufI[BLOCK_SIZE] ;
	sf_count_t frames ;
	int k, m, readcount ;

    float in2[BLOCK_SIZE], in2I[BLOCK_SIZE];
    fftwf_plan p, q;
    fftwf_plan pI, qI;
    fftwf_complex *out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * BLOCK_SIZE);
    fftwf_complex *outI = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * BLOCK_SIZE);

	frames = BLOCK_SIZE / channels ;

    
    

	while ((readcount = sf_readf_float (infile, buf, frames)) > 0)
	{	
        //Song
        p = fftwf_plan_dft_r2c_1d(BLOCK_SIZE, buf, out, FFTW_ESTIMATE);
        fftwf_execute(p);
        fftwf_destroy_plan(p);
        
        //Impulse
        sf_readf_float (impulsefile, bufI, frames);
        pI = fftwf_plan_dft_r2c_1d(BLOCK_SIZE, bufI, outI, FFTW_ESTIMATE);
        fftwf_execute(pI);
        fftwf_destroy_plan(pI);
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
            //out[i][0] = (out[i][0] * outI[i][0] - out[i][1] * outI[i][1]) * 1./BLOCK_SIZE;
            //out[i][1] = (out[i][0] * outI[i][1] + out[i][1] * outI[i][0]) * 1./BLOCK_SIZE;
            out[i][0] = (out[i][0] * outI[i][0] - out[i][1] * outI[i][1]) * 1./BLOCK_SIZE;
            out[i][1] = (out[i][0] * outI[i][1] + out[i][1] * outI[i][0]) * 1./BLOCK_SIZE;
        }

        //printf("\nInverse transform:\n");
        //Song
	    fprintf (outfile, "Inverse transform:\n");
        q = fftwf_plan_dft_c2r_1d(BLOCK_SIZE, out, in2, FFTW_ESTIMATE);
        fftwf_execute(q);
        // normalize 
        /*for (int i = 0; i < BLOCK_SIZE; i++) {
            in2[i] *= 1./BLOCK_SIZE;
        }*/
        
        //Impulse
        qI = fftwf_plan_dft_c2r_1d(BLOCK_SIZE, outI, in2I, FFTW_ESTIMATE);
        fftwf_execute(qI);
        // normalize 
        /*for (int i = 0; i < BLOCK_SIZE; i++) {
            in2I[i] *= 1./BLOCK_SIZE;
        }*/
        
        //Store data in file
        for (int i = 0; i < BLOCK_SIZE; i++) // Single channel
        {
            //in2[i] = buf[i] * bufI[i];  // For simple multiplication

            //in2I[i] *= in2[i];
            //printf("recover: %3d %+9.5f I vs. %+9.5f I\n", i, buf[i], in2[i]);
			//fprintf (outfile, "SSS:recover: %3d %+9.5f I vs. %+9.5f I\n", i, buf[i], in2[i]) ;
			//fprintf (outfile, "III:recover: %3d %+9.5f I vs. %+9.5f I\n", i, bufI[i], in2I[i]) ;
        }
        sf_write_float(finalfile, in2, frames);
        //sf_write_float(finalfile, in2I, frames);
        fftwf_destroy_plan(q);

        fftwf_cleanup();


        /*for (k = 0 ; k < readcount ; k++)
		{	
            for (m = 0 ; m < channels ; m++)
            {
					fprintf (outfile, " % 12.10f", buf [k * channels + m]) ;
            }
		}*/ 
	} 

	return ;
} /* convert_to_text */

int
main (int argc, char * argv [])
{	
    char 		*progname, *infilename, *outfilename, *impulsefilename, *finalfilename;
	SNDFILE		*infile = NULL ;
	FILE		*outfile = NULL ;
	SNDFILE		*impulsefile = NULL ;
	SNDFILE		*finalfile = NULL ;
	SF_INFO		sfinfo ;
	SF_INFO		sfinfoI ;
	SF_INFO		sfinfoO ;
	int		full_precision = 0 ;

	progname = strrchr (argv [0], '/') ;
	progname = progname ? progname + 1 : argv [0] ;

	switch (argc)
	{	case 6 :
			if (!strcmp ("--full-precision", argv [5]))
			{	print_usage (progname) ;
				return 1 ;
				} ;
			full_precision = 1 ;
			argv++ ;
		case 5 :
			break ;
		default:
			print_usage (progname) ;
			return 1 ;
		} ;

	infilename = argv [1] ;
    impulsefilename = argv[2];
    finalfilename = argv[3];
	outfilename = argv [4] ;


	if (strcmp (infilename, outfilename) == 0)
	{	printf ("Error : Input and output filenames are the same.\n\n") ;
		print_usage (progname) ;
		return 1 ;
		} ;

	if (infilename [0] == '-')
	{	printf ("Error : Input filename (%s) looks like an option.\n\n", infilename) ;
		print_usage (progname) ;
		return 1 ;
		} ;

	if (outfilename [0] == '-')
	{	printf ("Error : Output filename (%s) looks like an option.\n\n", outfilename) ;
		print_usage (progname) ;
		return 1 ;
		} ;
	
    if (impulsefilename[0] == '-')
	{	printf ("Error : Input impulse filename (%s) looks like an option.\n\n", impulsefilename) ;
		print_usage (progname) ;
		return 1 ;
		} ;
	
    if (finalfilename[0] == '-')
	{	printf ("Error : Final output filename (%s) looks like an option.\n\n", finalfilename) ;
		print_usage (progname) ;
		return 1 ;
		} ;

	memset (&sfinfo, 0, sizeof (sfinfo)) ;
	memset (&sfinfoI, 0, sizeof (sfinfoI)) ;
	memset (&sfinfoO, 0, sizeof (sfinfoO)) ;

	if ((infile = sf_open (infilename, SFM_READ, &sfinfo)) == NULL)
	{	printf ("Not able to open input file %s.\n", infilename) ;
		puts (sf_strerror (NULL)) ;
		return 1 ;
		} ;

	/* Open the output file. */
	if ((outfile = fopen (outfilename, "w")) == NULL)
	{	printf ("Not able to open output file %s : %s\n", outfilename, sf_strerror (NULL)) ;
		return 1 ;
		} ;

	if ((impulsefile = sf_open (impulsefilename, SFM_READ, &sfinfoI)) == NULL)
	{	printf ("Not able to open input file %s.\n", impulsefilename) ;
		puts (sf_strerror (NULL)) ;
		return 1 ;
		} ;

        sfinfoO.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
        sfinfoO.channels = sfinfo.channels;
        sfinfoO.samplerate = sfinfo.samplerate;


	if ((finalfile = sf_open (finalfilename, SFM_WRITE, &sfinfoO)) == NULL)
	{	printf ("Not able to open input file %s.\n", finalfilename) ;
		puts (sf_strerror (NULL)) ;
		return 1 ;
		} ;

	fprintf (outfile, "# Converted from file %s.\n", infilename) ;
	fprintf (outfile, "# Channels %d, Sample rate %d\n", sfinfo.channels, sfinfo.samplerate) ;

    clock_t start, end;
    double cpu_time_used;
    start = clock();
    //printf("Start = %d\n", start);
	convert_to_text (infile, impulsefile, finalfile, outfile, sfinfo.channels, full_precision) ;
    end = clock();
    //printf("End = %d\n", end);
    //cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    cpu_time_used = ((double) (end - start));
    printf("Time to Exec = %f\n", cpu_time_used);

	sf_close (infile) ;
	sf_close (impulsefile) ;
	sf_close (finalfile) ;
	fclose (outfile) ;

	return 0 ;
} /* main */

