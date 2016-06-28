#include "Input.h"
Input::Input()	: half_width(0.1),
		mean_mass(10.0),
		lorentz(0.1),
		temperature(296.0),
		partition(1000.0),
		pressure(1.0),
		nu_start(0),
		nu_end(100),
		Npoints(1000),
		num_files(0),
		gamma_air(0.05),
		n_air(0.4),
		hitran_mixture_air(1.0),
		max_points(500){

};

Input::~Input(){}

void Input::ReadInput(){
	string line;
	char* line_ptr;
	char linestr[1024];
	string spec_type;
	cin>>num_files;
	getline(cin,line);
	cout<<"line "<<line;
	//strcpy (linestr, line.c_str());
		//Begin Reading
	//line_ptr=strtok(linestr," (");
		//This should contai the first variable to check
	//num_files = readi();
	printf("num_files = %d\n",num_files);
	for(int i = 0; i < num_files; i++){
		cin>>line;
		cout<<line<<endl;
		trans_files.push_back(trim(line));
		getline(cin,line);
	}
	cin>>state_file;
	cout<<state_file<<endl; //state file
	//no of quanta and null
	getline(cin,line);
	getline(cin,line);
	getline(cin,line);
	cin >> temperature>>partition;
	//cout<<temperature<<" "<<partition<<endl;
	getline(cin,line);
	cin >> nu_start>>nu_end>>Npoints;
	//cout<<nu_start<<"  "<<nu_end<<" "<<partition<<endl;
	getline(cin,line);
	cin >> spec_type;
	cout<<spec_type<<endl;
	if(trim(spec_type).compare("gauss")==0){
		profile = GAUSSIAN;
	}else if(trim(spec_type).compare("doppl")==0){
		profile = DOPPLER;
	}else if(trim(spec_type).compare("voigt")==0){
		profile = VOIGT;
	}else{
		cout<<spec_type<<" not supported!"<<endl;
	}
	getline(cin,line);
	cin>>spec_type;
	getline(cin,line);
	if(trim(spec_type).compare("absorption")!=0){
		cout<<"Only absorption is supported\n"<<endl; 
	}
	if(profile == GAUSSIAN)
		cin>>half_width;
	else{
		cin>>mean_mass;
		getline(cin,line);
		cin>>half_width;		
	}
	





};

ProfileType Input::StringToProfile(string & pString){

	if(trim(pString)=="GAUSSIAN")
		return GAUSSIAN;
	else if(trim(pString)=="DOPPLER")
		return DOPPLER;
	else if(trim(pString)=="VOIGT")
		return VOIGT;
	else
		return PSUEDO_VOIGT;

}

void Input::ReadInputII(){
	//Read the first line

	stringstream s_cin;
	s_cin<<"HITRAN\nPARTITION -1.725235E+03\nN-POINTS 14001\nNU-RANGE 0.0 10000.0\nPROFILE VOIGT\nMEAN-MASS 17.031\nHITRAN-MIXTURE-AIR 1.0\nTEMPERATURE-PRESSURE-GRID\n296.0 0.0986923\nEND\nMAX-POINTS 14000001\nNUM-THREADS 0\nMEMORY 10\nTRANS-FILES\nNH3.par\nEND\n";
	ifstream i_cin("BT2_test_100_nvvp.inp");
	istream & input_stream = cin;

	string line;
	getline(input_stream,line);

	//Figure out if we need to deal with HITRAN or EXOMOL	
	if(trim(line)=="HITRAN"){
		which_file = HITRAN_TYPE;
	}else if(trim(line)=="EXOMOL"){
		which_file = EXOMOL_TYPE;
	}else{
		cout<<"File type: "<<trim(line)<<" not supported"<<endl;
	}
	
	partition = -1.0;
	num_threads=1;
	memory = 1000000000l;
	while(getline(input_stream,line)){
		//Clean the line
		line =trim(line);
		//Now lets split;
		vector<string> split_line = split(line);
		//Now lets check what we have
		if(split_line.size()==0) continue;
		string indent_string = 	trim(split_line[0]);
		
		if(indent_string == "PRESSURE")
			pressure=atof(split_line[1].c_str());
		//else if (indent_string == "TEMPERATURE")
		//	temperature=atof(split_line[1].c_str());
		//else if (indent_string == "TEMPERATURES"){
		//	while(getline(cin,line)){
		//		if(trim(line)=="END") break;
		//		TemperaturePressure t_temp_grid;
		//		t_temp_grid.temperature =atof(trim(line).c_str());
		//		temp_press_grid.push_back(t_temp_grid);
		//	}
		//}
		else if (indent_string == "TEMPERATURE-PRESSURE-GRID" || indent_string == "TEMPERATURE-HALFWIDTH-GRID" || indent_string == "TEMPERATURES"){
			while(getline(input_stream,line)){
				if(trim(line)=="END") break;
				vector<string> split_line_grid = split(line);

				TemperaturePressure t_temp_grid;
				t_temp_grid.temperature = atof(split_line_grid[0].c_str());
				if(split_line_grid.size()>1){
					t_temp_grid.pressure = atof(split_line_grid[1].c_str());	
				}else{
					t_temp_grid.pressure = 0.1;
				}		
				temp_press_grid.push_back(t_temp_grid);
			}
		}
		else if (indent_string == "MEAN-MASS")
			mean_mass=atof(split_line[1].c_str());
		else if (indent_string == "HITRAN-MIXTURE-AIR")
			hitran_mixture_air=atof(split_line[1].c_str());
		else if (indent_string == "PARTITION"){
			partition=atof(split_line[1].c_str());
			//if(partition<=0 && which_file==HITRAN_TYPE){
			//	printf("Partition computation of HITRAN files not implemented\n");
			//	exit(0);
			//}
		} 
		else if (indent_string == "HALF-WIDTH")
			half_width=atof(split_line[1].c_str()); 
		else if (indent_string == "GAMMA") 
			gamma_air=atof(split_line[1].c_str());
		else if (indent_string == "GAMMA-N")
			n_air=atof(split_line[1].c_str());
		else if (indent_string == "MAX-POINTS")
			max_points=atoi(split_line[1].c_str());
		else if (indent_string == "STATE-FILE"){
			
			state_file = trim(split_line[1]).c_str();
			cout<<"State File "<<state_file<<endl;
		}
		else if (indent_string == "TRANS-FILES"){
			while(getline(input_stream,line)){
				if(trim(line)=="END") break;
				trans_files.push_back(trim(line));
			}
		}else if (indent_string == "BROADENERS"){
			while(getline(input_stream,line)){
				if(trim(line)=="END") break;
				broadener_files.push_back(trim(line));
				
			}
		}else if(indent_string == "PROFILE"){
			profile = StringToProfile(trim(split_line[1]));
		}else if(indent_string == "N-POINTS"){
			Npoints = atoi(split_line[1].c_str());
		}else if(indent_string == "NU-RANGE"){
			nu_start = atof(split_line[1].c_str());
			nu_end = atof(split_line[2].c_str());
		}else if(indent_string == "MEMORY"){
			size_t mem = atoi(split_line[1].c_str());
			memory = mem*1000000000l;
		}else if(indent_string == "NUM-THREADS"){
			num_threads= atoi(split_line[1].c_str());
		}	
			  
		
		



	}

	if(broadener_files.size()==0)
		broadener_files.push_back("");

	std::cout<<"-----------------Settings-----------------"<<std::endl;
	if(temp_press_grid.size()>0){
		cout<<"---------Temperature pressure grid---------"<<endl;
		for(int i = 0; i < temp_press_grid.size(); i++)
			cout<<"Temperature: "<<temp_press_grid[i].temperature<<" K  Pressure: "<<temp_press_grid[i].pressure<<" atm"<<endl;
	}else if(temperatures.size()>0){
		cout<<"--------Temperatures-------------"<<endl;
		for(int i = 0; i < temperatures.size(); i++)
			cout<<"Temperature: "<<temperatures[i]<<" K"<<endl;
	}else{
		cout<<"Temperature: "<<temperature<<" K  Pressure: "<<pressure<<" atm"<<endl;
	}
	cout<<"Partition: "<<partition<<" HITRAN Mix "<<hitran_mixture_air<<endl;
	cout<<"Half-Width: "<<half_width<<endl;
	cout<<" Nu Range: "<<nu_start<<" - "<<nu_end<<endl;
	cout<<" Npoints: "<<Npoints<<endl;
	

	

	

	

};








