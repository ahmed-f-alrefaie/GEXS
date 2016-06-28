#include "BaseProfile.h"

BaseProfile::BaseProfile(Input* pInput): input(pInput), Ntrans(0), h_energies(NULL), h_nu(NULL),h_gns(NULL), h_gammaL(NULL),h_n(NULL),h_aif(NULL), total_intens(0.0) {



	start_nu = input->GetNuStart();
	end_nu = input->GetNuEnd();
	Npoints = input->GetNpoints();
	dfreq = (end_nu-start_nu)/double(Npoints);
	printf("Start_nu = %12.6f End_nu = %12.6f Dfreq = %12.6f Npoints =%d\n",start_nu,end_nu,dfreq,Npoints);

	freq = new double[Npoints];
	intens= new double[Npoints];


	for(int i = 0; i < Npoints; i++)
	{
		freq[i]=start_nu+double(i)*dfreq;
		intens[i]=0.0;
	}

	temperature_pressure_comp = pInput->GetTemperaturePressureGrid();
	num_intens = temperature_pressure_comp.size();
	
	//GetTemperaturePressureGrid(){return temp_press_grid;}
	//vector<double> GetTemperatures(){return temperatures;}


}

BaseProfile::~BaseProfile(){

	delete[] freq;
	delete[] intens;

	


}

void BaseProfile::Initialize(){
	profile = GetProfile();
	double ref_temp;	
	double ref_press=1.0;
	//Initialize the stateReader TODO: Use Factory Design Pattern for this
	if(input->GetFileType()==HITRAN_TYPE){
		state_reader = (StateReader*)(new HITRANStateReader( input->GetTransFiles()[0] , input->GetPartition(),input->GetPressure(),input->GetTemperature(),input->GetHitranMixture()   ));
		ref_temp = 296.0;
		ref_press = 1.0;
	}else if(input->GetFileType()==EXOMOL_TYPE){
		state_reader = (StateReader*)(new ExomolStateReader( input->GetStateFile() , input->GetPartition(),input->GetPressure() ,input->GetBroadeners()[0],input->GetGamma(),input->GetGammaN()));
		ref_temp = 256.0;
		ref_press= 1.0/0.986923267;	
	//	exit(0);
	}else{
		exit(0);
	}

	//manager=new HybridManager(profile,input->GetNumThreads(),input->GetMemory());
	manager=new MultiGpuManager(profile);
	ComputePartitions();


	manager->InitializeVectors(Npoints,num_intens);

	manager->TransferFreq(freq,intens,Npoints);
	num_trans_fit=manager->GetNtrans();


	manager->InitializeConstants(input->GetHalfWidth(),input->GetTemperature(), state_reader->ComputePartition(input->GetTemperature()),dfreq,input->GetMeanMass(),input->GetPressure(),ref_temp,ref_press);


	h_energies = new double[num_trans_fit];
	h_nu = new double[num_trans_fit];
	h_aif = new double[num_trans_fit];
	h_gns = new int[num_trans_fit];
	if(require_pressure){	
		h_gammaL = new double [num_trans_fit];
		h_n = new double[num_trans_fit];
	}

	//InitializeProfile();	
	//Get our transition files
	transition_files = input->GetTransFiles();
	
	

}

void BaseProfile::ComputePartitions(){
	//if(num_intens>1){
	printf("Ccmputing partitions:\n");
	for(int i = 0; i < temperature_pressure_comp.size(); i++){
		temperature_pressure_comp[i].partition = state_reader->ComputePartition(temperature_pressure_comp[i].temperature);
		printf("T: %12.6f K Pr: %12.6f atm Q: %12.6f\n",temperature_pressure_comp[i].temperature,temperature_pressure_comp[i].pressure,temperature_pressure_comp[i].partition);
	}
}

void BaseProfile::PerformCrossSection(double HW){
		printf("Waiting for previous calculation to finish......\n");
		fflush(0);
		Timer::getInstance().StartTimer("Execute Cross-Section");	

		manager->TransferVectors(Ntrans,h_energies, h_nu, h_aif,h_gns,h_gammaL,h_n);
		ib = std::max(round( ( min_nu-HW-start_nu)/dfreq ),0.0);
		ie =  std::min(round( ( max_nu+HW-start_nu)/dfreq ),double(Npoints));
		points = ie - ib;
		printf("Min nu = %12.6f Max_nu = %12.6f ib = %d ie = %d points = %d\n",min_nu,max_nu,ib,ie,points);
		//fflush(0);
		for(int i = 0; i < temperature_pressure_comp.size(); i++)
				manager->ExecuteCrossSection(points, Ntrans,ib,temperature_pressure_comp[i].temperature,temperature_pressure_comp[i].pressure,temperature_pressure_comp[i].partition);
		Timer::getInstance().EndTimer("Execute Cross-Section");		
		ComputeTotalIntensity();
		
}


void BaseProfile::ExecuteCrossSection(){
	//We have no transitions right now	
	Ntrans = 0;
	double t_nu,t_ei,t_aif,t_gns,t_gammaL,t_n;

	int num_trans_files = transition_files.size();
	max_nu = 0.0;
	min_nu = 99999.0;
	double gammaHW = m_half_width;
	int current_Ntrans=0;
	//double gammaHW = std::min(25.0*input->GetPressure(),100.0);
	int temp_ib;
	int temp_ie;
	int temp_points;
	int max_points = input->GetMaxPoints();
	max_points = std::max((int)((gammaHW)*0.1/dfreq),max_points);
	max_points /= temperature_pressure_comp.size();
	max_points+=1;
	int def_max_points = max_points;

	//Compute the beta for the first temperature
	m_beta = PLANCK*VELLGT/(BOLTZ*temperature_pressure_comp[0].temperature);
	int runs=0;
	printf("Range = %12.6f - %12.6f %d\n",start_nu-gammaHW,end_nu+gammaHW,max_points);
	for(int  i = 0; i < num_trans_files; i++){
		state_reader->OpenFile(transition_files[i]);
		printf("%s\n",transition_files[i].c_str());
		fflush(0);
		Timer::getInstance().StartTimer("FileIO");
		while(state_reader->ReadNextState(t_nu,t_gns,t_ei,t_aif,t_gammaL,t_n)){


			
			if(t_nu< start_nu-gammaHW)
				continue;
			if(t_nu> end_nu+gammaHW)
				break;
			if(t_nu ==0.0)
				continue;
			h_energies[Ntrans] = t_ei;
			h_nu[Ntrans] = t_nu;
			h_gns[Ntrans] = t_gns;
			h_aif[Ntrans] = t_aif;
			if(require_pressure){
				h_gammaL[Ntrans] = t_gammaL;
				h_n[Ntrans] = t_n;	
			}				
			
			
			//Find the maximum and minimum nu 
			min_nu=std::min(min_nu,h_nu[Ntrans]);
			max_nu=std::max(max_nu,h_nu[Ntrans]);		
			//max_gammaL = std::max(max_gammaL,h_gammaL[Ntrans]);
			//min_n = std::min(min_n,h_n[Ntrans]); 
			//Restrict to a set number of points
			temp_ib = std::max(round( ( min_nu-gammaHW-start_nu)/dfreq ),0.0);
			temp_ie =  std::min(round( ( max_nu+gammaHW-start_nu)/dfreq ),double(Npoints));
			temp_points = temp_ie - temp_ib;
			
			//if(
			Ntrans++;
			//this ensures the minimum work required is performed
			if((Ntrans>=num_trans_fit  || ( ( temp_points > max_points ) /*&& ( manager->ReadyForWork() )*/ ))  )
			{
				
				/*runs++;

				//This algortihm allows us to scale the computed resolution based on how quickly we can do them
				//It is based off the article "Dynamic Resolution Rendering Article" by Intel for rendering graphics
				float avg_time = Timer::getInstance().GetAvgTimeInSeconds("Execute Cross-Section");
				//Our scaling constant
				//Wait for three runs before tuning
				if(runs>2 && avg_time > 0.0f){
				double k = 1.1;
				float maxpointratio = (float)max_points/(float)def_max_points;
				
				//We want to hit 1 second
				float deltamaxpoints = k*maxpointratio*(1.0 - avg_time);
				maxpointratio += deltamaxpoints;
				
				max_points = (int)((float)max_points*maxpointratio);

				printf("Maxpoints = %d Min_maxpoints = %d\n",max_points, def_max_points);
				}
				*/
				PerformCrossSection(gammaHW);
				//float last_time = Timer::getInstance().GetRunningAvgTimeInSeconds("Execute Cross-Section");
				//If we are taking to long then reduce the number of gridpoints
				/*if(last_time > 1.0){
					printf("Decreasing points last_time = %12.6f\n",last_time);
					max_points = (int)(((float)max_points)/1.2f) + 1;
					printf("max_points = %d\n",max_points);
				}
				//If we arent taking enough time then increase the number of gridpoints if we havent reached memory limit
				else if(Ntrans < num_trans_fit && runs>3 && last_time > 0.0 && last_time < 0.5){
					printf("Increasing points last_time = %12.6f\n",last_time);
					max_points = (int)(((float)max_points)*1.1f) + 1;
					printf("max_points = %d\n",max_points);
				}
				
				runs++;
				*/
				//Check the last bit of 
				//printf("%d\n",temp_points);
				//gammaHW = 500.0*max_gammaL*pow((300.0/input->GetTemperature()),min_n)*input->GetPressure(); 
				


				//Reset counters		
				max_nu = 0.0;
				min_nu = 99999.0;
				Ntrans = 0;
			}
		}
		Timer::getInstance().EndTimer("FileIO");
		state_reader->CloseFile();

	}
	printf("Left over transitions: %d\n",Ntrans);
	if(Ntrans > 0 ){

		//gammaHW = 500.0*max_gammaL*pow((300.0/input->GetTemperature()),min_n)*input->GetPressure(); 
		PerformCrossSection(gammaHW);

				//Reset counters		
	}



}

void BaseProfile::ComputeTotalIntensity(){
		//Only deal with the first temperature
		double temp = temperature_pressure_comp[0].temperature;
		double partition = temperature_pressure_comp[0].partition;
		double temp_intens = 0.0;
		m_beta = PLANCK*VELLGT/(BOLTZ*temp);
		#pragma omp parallel for reduction(+:temp_intens)
		for(int i = 0; i < Ntrans; i++){
			double abscoef= CM_COEF*h_aif[i]*h_gns[i]
				*exp(-m_beta*h_energies[i])*(1.0-exp(-m_beta*h_nu[i]))/
				(h_nu[i]*h_nu[i]*partition);
			temp_intens+=abscoef;
		}

	total_intens+=temp_intens;


}

void BaseProfile::OutputProfile(){
	//Output Data
	/*if(num_intens==1){
		manager->TransferResults(freq,intens,Npoints);
		for(int j = 0; j < Npoints; j++)
			printf("##0 %12.6f %13.8E\n",freq[j],intens[j]);
	}else{
	*/

	for(int i = 0; i < num_intens; i++){
		for(int j = 0; j < Npoints; j++)
			intens[j] = 0.0;
		manager->TransferResults(freq,intens,Npoints,i);
		//Clear
		if (i ==0){
			double cross_intens = 0.0;
			#pragma omp parallel for reduction(+:cross_intens)
			for(int l =0; l < Npoints; l++){
				cross_intens+= intens[l];
			}
			cross_intens*=dfreq;
			printf("Our summed intensity = %16.6E compared to the integrated intensity = %16.6E with a difference of %12.6f % \n",total_intens,cross_intens,(total_intens-cross_intens)*100.0/total_intens);
		}
		printf("-------T=%12.6f K--------P=%12.6f atm---------\n\n",temperature_pressure_comp[i].temperature,temperature_pressure_comp[i].pressure);
		for(int j = 0; j < Npoints; j++)
			printf("[%i] %12.6f %13.8E\n",i,freq[j],intens[j]);
		printf("\n");
		
	}
	//}


}
