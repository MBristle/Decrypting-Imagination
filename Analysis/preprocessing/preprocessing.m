%% construct path
clear
addpath('FuncSimToolbox','ScanMatch','LIBSVM')

f=struct();
f.parts = strsplit(pwd, '/');
f.DirPart = fullfile( '/',f.parts{1:end-2},'EyetrackingData/');
f.d =dir(f.DirPart);

f.s = regexpi({f.d.name},'SampleReport\w*.xls','match');
f.s = [f.s{:}];

f.f = regexpi({f.d.name},'FixationReport\w*.xls','match');
f.f = [f.f{:}];

%% Read data
data=struct();
for i = 1:length(f.s)
   % data.s{i}= struct2table(tdfread([f.DirPart,f.s{1}],'\t'));
end
for i = 1:length(f.f)
    data.f{i}= struct2table(tdfread([f.DirPart,f.f{1}],'\t'));
end

%data.s=vertcat(data.f{:});
data.f=vertcat(data.f{:});

%% 
 %ToDo: Split set in to imagery and perception
 %
 %perform RCA, ScanMatch, Multimatch, FuncSim for each trail in each session for
 %each participant
 
 %Assumes for all participants the same number of trails in each session
 %and the same number of sessions for each participant! (see assertation)
 
OUT = regexp(table2cell(data.f(:,1)),...
    '^(?<participant>[a-zA-Z][a-zA-Z]\d\d)_(?<session>\d)$', 'names');
data.f= [data.f,struct2table([OUT{:}])];

%split set into imagery and perception
data.p.f = data.f(15000>data.f.CURRENT_FIX_START,:);
data.i.f = data.f(15000<data.f.CURRENT_FIX_START,:);

%get metadata about the set
data.md.trails=unique(data.p.f.TRIAL_INDEX);
data.md.sessions=unique(data.p.f.session);
data.md.participants=unique(data.p.f.participant);

%parameters of the RQA
param.delay = 1;
param.embed = 1;
param.rescale = 0;
param.metric = 'euclidian';
param.adjacency=[];
param.linelength = 2;
param.radius=64;

 for i_p =1:length(data.md.participants)
    curr_p=data.md.participants(i_p);
    assert(length(unique(data.p.f.session(strcmp(data.p.f.participant,curr_p),:)))...
        ==length(data.md.sessions),...
        char(strcat('Number of sessions does not match in "',curr_p,'"')))

    for i_s= 1:length(data.md.sessions)
        curr_s=data.md.sessions(i_s);
        assert(0.5*(length(unique(data.p.f.TRIAL_INDEX(strcmp(data.p.f.session,curr_s)&...
            strcmp(data.p.f.participant,curr_p))))...
             + length(unique(data.i.f.TRIAL_INDEX(strcmp(data.i.f.session,curr_s)&...
             strcmp(data.i.f.participant,curr_p)))))...
             ==length(data.md.trails),...
            char(strcat('Number of trails does not match in participant "',curr_p,...
            '" in session "',curr_s,'"')))
        
        for i_t=1:length(data.md.trails)
            curr_t=data.md.trails(i_t);
            
            
            % RQA-analysis
        
            results.p.rqa{i_p,i_s,i_t} =Rqa(...
                    [data.p.f.CURRENT_FIX_X(strcmp(data.p.f.session,curr_s)&...
                        strcmp(data.p.f.participant,curr_p)&data.p.f.TRIAL_INDEX==curr_t),...
                    data.p.f.CURRENT_FIX_Y(strcmp(data.p.f.session,curr_s)&...
                        strcmp(data.p.f.participant,curr_p)&data.p.f.TRIAL_INDEX==curr_t)]...
                ,param);

            
            results.i.rqa{i_p,i_s,i_t} =Rqa(...
                    [data.i.f.CURRENT_FIX_X(strcmp(data.i.f.session,curr_s)&...
                        strcmp(data.i.f.participant,curr_p)&data.i.f.TRIAL_INDEX==curr_t),...
                    data.i.f.CURRENT_FIX_Y(strcmp(data.i.f.session,curr_s)&...
                        strcmp(data.i.f.participant,curr_p)&data.i.f.TRIAL_INDEX==curr_t)]...
                ,param);
            
            % scanMatch analysis
                %TODO
                results.p.sm{i_p,i_s,i_t}=nan;
                results.i.sm{i_p,i_s,i_t}=nan;
            
            % multiMatch analysis
                %TODO
                results.p.mm{i_p,i_s,i_t}=nan;
                results.i.mm{i_p,i_s,i_t}=nan;
            
            % FuncSim analysis
                %TODO
                results.p.fs{i_p,i_s,i_t}=nan;
                results.i.fs{i_p,i_s,i_t}=nan;
                
            %  OTHER MEASUREMENT
                %TODO
                %results.p.REPLACE{i_p,i_s,i_t}=nan;
                %results.i.REPLACE{i_p,i_s,i_t}=nan;

            
        end
    end
 end
 %% Create Dataset 
 k=1;
 
  for i_p =1:length(data.md.participants)
    curr_p=data.md.participants(i_p);
    for i_s= 1:length(data.md.sessions)
        curr_s=data.md.sessions(i_s);
        for i_t=1:length(data.md.trails)
            curr_t=data.md.trails(i_t);
            curr_select_p=strcmp(data.p.f.session,curr_s)&...
                strcmp(data.p.f.participant,curr_p)&data.p.f.TRIAL_INDEX==curr_t;
            curr_select_i=strcmp(data.i.f.session,curr_s)&...
                strcmp(data.i.f.participant,curr_p)&data.i.f.TRIAL_INDEX==curr_t;

           
            
            % Create Dataset
            
            dataset.all{k}= {char(curr_p),... %Participant
                str2double(curr_s{:})==1,...%Session
                table2array(unique(data.p.f(curr_select_p,13))),... %Block
                unique(data.p.f.number(curr_select_p)),... % PictureID
                unique(data.p.f.TRIAL_INDEX(curr_select_p)),... % TrailIndex
                length(unique(data.p.f.CURRENT_FIX_INDEX(curr_select_p))),... % Total Fixation Perception
                length(unique(data.i.f.CURRENT_FIX_INDEX(curr_select_i))),... % Total Fixation Imagination
                table2array(unique(data.p.f(curr_select_p,14))),... % category
                table2array(unique(data.p.f(curr_select_p,12))),... %Rating
                num2cell(table2array(data.p.f(curr_select_p,5))'),... %Pupile - Perception
                num2cell(table2array(data.p.f(curr_select_p,2))'),... %Duration - Perception
                num2cell(table2array(data.p.f(curr_select_p,7))'),... %X - Perception
                num2cell(table2array(data.p.f(curr_select_p,8))'),... %Y - Perception
                results.p.rqa{i_p,i_s,i_t},... RQA - Perception 
                num2cell(table2array(data.i.f(curr_select_i,5))'),... %Pupile - Imagination
                num2cell(table2array(data.i.f(curr_select_i,2))'),... %Duration - Imagination
                num2cell(table2array(data.i.f(curr_select_i,7))'),... %X - Imagination
                num2cell(table2array(data.i.f(curr_select_i,8))')... %Y - Imagination
                results.i.rqa{i_p,i_s,i_t},... RQA - Imagination
                };
            
            k=k+1;
        end
    end
 end
dataset.all=vertcat(dataset.all{:});






