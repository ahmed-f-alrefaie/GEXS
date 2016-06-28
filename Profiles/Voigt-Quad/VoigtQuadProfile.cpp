#include "VoigtProfile.h"
#include "Timer.h"

void VoigtQuadProfile::InitializeProfile(){
	profile = VOIGTQUAD;
	double ref_temp;	
	printf("Profile Selected is Voigt\n");
	//Initialize the stateReader TODO: Use Factory Design Pattern for this
	if(input->GetFileType()==HITRAN_TYPE){
		state_reader = (StateReader*)(new HITRANStateReader( input->GetTransFiles()[0] , input->GetPartition(),input->GetPressure(),input->GetTemperature(),input->GetHitranMixture()   ));
		ref_temp = 296.0;
	}else if(input->GetFileType()==EXOMOL_TYPE){
		state_reader = (StateReader*)(new ExomolStateReader( input->GetStateFile() , input->GetPartition(),input->GetPressure() ,input->GetBroadeners()[0]));
		ref_temp = 500.0;	
	//	exit(0);
	}else{
		exit(0);
	}

	manager=new HybridManager(profile,input->GetNumThreads(),input->GetMemory());
	manager->InitializeVectors(Npoints);
	manager->TransferFreq(freq,intens,Npoints);
	num_trans_fit=manager->GetNtrans();


	manager->InitializeConstants(input->GetHalfWidth(),input->GetTemperature(), state_reader->ComputePartition(input->GetTemperature()),dfreq,input->GetMeanMass(),input->GetPressure(),ref_temp);

	
	h_energies = new double[num_trans_fit];
	h_nu = new double[num_trans_fit];
	h_aif = new double[num_trans_fit];
	h_gns = new int[num_trans_fit];
	h_gammaL = new double [num_trans_fit];
	h_n = new double[num_trans_fit];
	
}

void VoigtQuadProfile::ExecuteCrossSection(){
	//We have no transitions right now	
	Ntrans = 0;
	double t_nu,t_ei,t_aif,t_gns,t_gammaL,t_n;

	int num_trans_files = transition_files.size();
	max_nu = 0.0;
	min_nu = 99999.0;
	max_gammaL=0.0;
	min_n=99999.0;
	int current_Ntrans=0;
	double gammaHW = std::min(25.0*input->GetPressure(),100.0);
	int temp_ib;
	int temp_ie;
	int temp_points;
	int max_points = input->GetMaxPoints();

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

			h_energies[Ntrans] = t_ei;
			h_nu[Ntrans] = t_nu;
			h_gns[Ntrans] = t_gns;
			h_aif[Ntrans] = t_aif;
			h_gammaL[Ntrans] = t_gammaL;
			h_n[Ntrans] = t_n;					
			
			
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
			if(Ntrans>=num_trans_fit || temp_points > max_points){
				//printf("%d\n",temp_points);
				//gammaHW = 500.0*max_gammaL*pow((300.0/input->GetTemperature()),min_n)*input->GetPressure(); 
				PerformCrossSection(gammaHW);


				//Reset counters		
				max_nu = 0.0;
				min_nu = 99999.0;
				max_gammaL = 0.0;
				min_n = 99999.0;
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

