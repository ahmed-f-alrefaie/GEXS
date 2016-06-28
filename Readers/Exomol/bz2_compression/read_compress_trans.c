#include "bzlib.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>



char* unusedTmp;
char output_buff[38];
char unused[5000];
FILE* transition_file=NULL;
BZFILE* bz_trans=NULL;	
int bz2_error;
int nUnused = 0;
int nRead = 0;
int streamNo=0;
char* unusedtmp;
int stream_count = 0;

void open_transition_file_bz2_(const char* filename){
	transition_file=fopen(filename,"rb");
	if(transition_file == NULL){
		printf("Error opening file %s\n",filename);
	}
	output_buff[37]='\0';
	nUnused = 0;
	nRead = 0;
	streamNo=0;
	stream_count = 0;	
	bz_trans = BZ2_bzReadOpen(&bz2_error,transition_file,0,0,unused,nUnused);

	//printf("Here3");
};

void read_trans_lines_bz2_(int* id_f,int* id_i,double* aif,int* error){
	if(transition_file == NULL){
		*error = 1;
		return;
	}
	if (nUnused == 0 && feof(transition_file)>0 && bz2_error == BZ_STREAM_END) {
		printf("Finished! %d\n",bz2_error);
		BZ2_bzReadClose ( &bz2_error, bz_trans );
		bz_trans = NULL;
		*error = 1;
		return;
	}
	int new_pos = 0;
	nRead = BZ2_bzRead ( &bz2_error, bz_trans, output_buff, 37 );
	//printf("Error = %d nRead = %d",bz2_error,nRead);
	//printf("Here5 %p %p",(void*)bz_trans,transition_file);
	if ((bz2_error == BZ_OK || bz2_error == BZ_STREAM_END) && nRead > 0){
		char* ln_ptr;
		//If we havent finished reading the full line the we need to move
	
			
		if(bz2_error == BZ_STREAM_END){
				//exit(0);
				//printf("\n\nCut stream here\n\n");
				BZ2_bzReadGetUnused ( &bz2_error, bz_trans, (void**)&unusedtmp, &nUnused );
				//printf("\n\nUnsued: %d \n\n",nUnused);
				int i;
				for(i = 0; i < nUnused; i++) unused[i]=unusedtmp[i];
				BZ2_bzReadClose ( &bz2_error, bz_trans );
				bz_trans = NULL;
				//printf("%s   read:%d",output_buff,nRead);
				//printf("%s",);
				bz_trans = BZ2_bzReadOpen(&bz2_error,transition_file,0,0,unused,nUnused);
				//nRead = BZ2_bzRead ( &bz2_error, bz_trans, output_buff, 37 );
				//The reading needs to be ofset to account for the miss reading

				
			
		}
		if(nRead<37){
					//printf("after cut nRead = %d\n",nRead);
					nRead = BZ2_bzRead ( &bz2_error, bz_trans, output_buff+nRead, 37-nRead );
					if(bz2_error != BZ_OK || bz_trans == NULL){
						printf("Error opening file for BZ2\n");
					}
					//if(bz2_error == BZ_STREAM_END && nRead==0) {*error=1; return;}
					//printf("\n\nCut stream here unused\n\n");
		}
		
		//printf("%s",output_buff);
		*id_f = strtol(output_buff,&ln_ptr,0);
		*id_i = strtol(ln_ptr,&ln_ptr,0);
		*aif = strtod(ln_ptr,&ln_ptr);
		*error=0; return;
	}
	//printf("Here6");
		
	//if(nRead==0){*error=1; return;} 
	
};

void close_transition_file_bz2_(){
	if(bz_trans != NULL){
		BZ2_bzReadClose ( &bz2_error, bz_trans );
		bz_trans = NULL;
	}
	if(transition_file != NULL){
		fclose(transition_file);
		transition_file = NULL;
	}

};



