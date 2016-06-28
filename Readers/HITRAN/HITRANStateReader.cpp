#include "HITRANStateReader.h"

int GetFieldStart(int field){
	const int field_lengths[] =  {2,1,12,10,10,5,5,10,4,8,15,15,15,15,6,12,1,7,7};
	int start = 0;	
	for(int i = 0; i < field; i++){
		start=start+field_lengths[i];
	} 
		
	return start;
	
};

int GetFieldLength(int field){
	const int field_lengths[] =  {2,1,12,10,10,5,5,10,4,8,15,15,15,15,6,12,1,7,7};
	return field_lengths[field];
	
};


void GetField(int field_num,const char* buffer,double & f){
	char c_f[13]={0,0,0,0,0,0,0,0,0,0,0,0,0};
	int start = GetFieldStart(field_num);
	int length = GetFieldLength(field_num);
	memcpy(c_f,buffer+start,length);	
	c_f[12]='\0';
	f = atof(c_f);
	
};
void GetField(int field_num,const char* buffer,int & f){
	char c_f[13]={0,0,0,0,0,0,0,0,0,0,0,0,0};
	int start = GetFieldStart(field_num);
	int length = GetFieldLength(field_num);

	memcpy(c_f,buffer+start,length);	
	c_f[12]='\0';
	//printf("start %d end %d   : %s\n",start,length,c_f);
	f = atoi(c_f);
};

HITRANStateReader::HITRANStateReader(std::string pFilename,double pPartition,double pressure,double temperature,double mixture):StateReader(pFilename,pPartition) , m_pressure(pressure),m_mixture(mixture), m_temperature(temperature){
	hitran_file = fopen(m_filename.c_str(),"r");
	if(hitran_file==NULL){
		printf("Could not open file %s ",m_filename.c_str());
		//return false;
		exit(0);
	}
	//open_file = true;

	//Get molecular id and iso number
	if(NULL==fgets(buffer,1024,hitran_file)){
		printf("HITRAN FILE IS EMPTY!!");
		exit(0);
	}
	
	GetField(0,(char*)buffer,mol_id);
	GetField(1,(char*)buffer,iso_num);
	rewind(hitran_file); //Move back now
	printf("Mol ID: %d , Iso: %d\n",mol_id,iso_num);
	double d_gi=0.0;
	fclose(hitran_file);
	if(partition<=0)bd_tips_2011_(&mol_id,&temperature,&iso_num,&d_gi,&partition);


}

bool HITRANStateReader::OpenFile(std::string pFilename){
	if(pFilename == "")
		return false;
	if(open_file==true){
		CloseFile();
	}
	
	hitran_file = fopen(m_filename.c_str(),"r");
	if(hitran_file==NULL){
		printf("Could not open file %s ",m_filename.c_str());
		return false;
	}
	open_file = true;

	//Get molecular id and iso number
	if(NULL==fgets(buffer,1024,hitran_file)){
		printf("HITRAN FILE IS EMPTY!!");
		exit(0);
	}
	
	GetField(0,(char*)buffer,mol_id);
	GetField(1,(char*)buffer,iso_num);
	rewind(hitran_file); //Move back now
	//printf("Mol ID: %d , Iso: %d\n",mol_id,iso_num);

	double d_gi = 0.0;
	double d_temp = 296.0;
	//Get the reference temperature
	bd_tips_2011_(&mol_id,&d_temp,&iso_num,&d_gi,&ref_partition);
	printf("Reference partition: %12.6f\n",ref_partition);

	return true;
	
}

bool HITRANStateReader::CloseFile(){
	if(open_file==true && hitran_file != NULL){
		fclose(hitran_file);
	}


}

bool HITRANStateReader::ReadNextState(double & nu,double & gns,double & e_i, double & aif){

	double dummy_gam,dummy_n;
	return ReadNextState(nu,gns,e_i,aif,dummy_gam,dummy_n);

	
	


}

double HITRANStateReader::ComputePartition(double temperature){
	double d_gi=0.0;
	//if(partition <= 0){
		//printf("\n computing....");
		//double d_temp =temperature 
		//Recompute
		//partition = 0.0;
		bd_tips_2011_(&mol_id,&temperature,&iso_num,&d_gi,&partition);
		printf("%12.6f Partition computed for temp %12.6f is : %12.6f\n",d_gi,temperature,partition);
		
	//}
	
	return partition;

}

bool HITRANStateReader::ReadNextState(double & nu,double & gns,double & e_i, double & aif, double & gam,double & n){

	if(!open_file)
		return false;
	/*
	char c_nu[13];
	char c_eif[13];
	char c_gns[13];
	char c_aif[13];
	char c_gam[13];
	char c_n[13];
	char c_self[13];
	*/

	//Replace soon
	double gam_air;
	double self;
	double intens;
	//printf("%s\n",buffer);
	if(NULL==fgets(buffer,1024,hitran_file))
		return false;
	
	//for(int i = 0; i < 160; i++)
	//	if(buffer[i]==' ') buffer[i]='0';

//	scanf(buffer,"%2u%1u%12f%10f%10f%5f%5f%10f%4f%8f%15c%15c%15c%15c%6c%12c%1c%7f%7f",
//&d1,&d1,&nu,&dd1,&aif,&dd1,&dd1,&e_i,&dd1,&dd1,small_buff,small_buff,small_buff,
//	small_buff,small_buff,&d1,&d1,small_buff,&gns,&dd1);
	/*
	int start = GetFieldStart(2);
	int length = GetFieldLength(2);


	memcpy(c_nu,buffer+start,length);

	start = GetFieldStart(4);
	length = GetFieldLength(4);
	memcpy(c_aif,buffer+start,length);	
		
	start = GetFieldStart(7);
	length = GetFieldLength(7);
	memcpy(c_eif,buffer+start,length);	

	start = GetFieldStart(5);
	length = GetFieldLength(5);
	memcpy(c_gam,buffer+start,length);	

	start = GetFieldStart(6);
	length = GetFieldLength(6);
	memcpy(c_self,buffer+start,length);	

	start = GetFieldStart(8);
	length = GetFieldLength(8);
	memcpy(c_n,buffer+start,length);	

	start = GetFieldStart(17);
	length = GetFieldLength(17);
	memcpy(c_gns,buffer+start,length);

	

	c_nu[12]='\0';
	c_eif[12]='\0';
	c_aif[12]='\0';
	c_gns[12]='\0';
	c_gam[12]='\0';
	c_n[12] = '\0';
	c_self[12] = '\0';
	//printf("Here: %s %14.3E\n",c_nu,atof(c_nu));

	*/
	
	GetField(2,(char*)buffer,nu);
	GetField(7,(char*)buffer,e_i);
	GetField(4,(char*)buffer,aif);
	GetField(5,(char*)buffer,gam_air);
	GetField(6,(char*)buffer,self);
	GetField(8,(char*)buffer,n);
	GetField(17,(char*)buffer,gns);
	GetField(3,(char*)buffer,intens);
	//printf("Mixture ratio.......%12.6f\n",m_mixture);
	gam = (gam_air*m_mixture + self*(1.0-m_mixture))*m_pressure;
	//printf("%12.6f %12.6f %14.4E %d %12.6f %12.6f %12.6f \n",nu, e_i, aif, int(gns),gam,n,self);
	//n=atof(c_n);

	if(aif==0.0) {
		double planck =6.6260693e-27;
		double avogno=6.0221415e23;
		double vellgt=2.99792458e10;
		double boltz=1.380658e-16;
		double pi=std::acos(-1.0);
		double beta=planck*vellgt/(boltz*296.0);
		//GNSAIF
		//Compute the 'GNSAIF'
		aif = intens*8.0*pi*vellgt*ref_partition*nu*nu/(exp(-beta*e_i)*(1.0-exp(-beta*nu)));
		gns=1.0;

		//printf("NEW: %14.4E \n",aif);
		/*
		start = GetFieldStart(3);
		length = GetFieldLength(3);
		memcpy(c_aif,buffer+start,length);	
		c_aif[12]='\0';
		//printf("%s\n",buffer);
		//printf("%12.6f %12.6f %14.4E %d %12.6f %12.6f\n",nu, e_i, aif, int(gns),gam,n);
		aif = atof(c_aif);
		gns=-1.0;
		*/
		//
		
	}
	return true;

}

HITRANStateReader::~HITRANStateReader(){
	CloseFile();
}
