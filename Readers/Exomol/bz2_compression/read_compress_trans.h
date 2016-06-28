#ifndef READ_TRANS_H
#define READ_TRANS_H



extern "C" { 
	void open_transition_file_bz2_(const char* filename);
	void read_trans_lines_bz2_(int* id_f,int* id_i,double* aif,int* error);
	void close_transition_file_bz2_();
};


#endif
