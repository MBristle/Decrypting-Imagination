%% construct path
clear
addpath('FuncSimToolbox','ScanMatch','LIBSVM')

f=struct();
f.parts = strsplit(pwd, filesep);
f.DirPart = fullfile(filesep,f.parts{1:end-2},['EyetrackingData',filesep]);
f.d =dir(f.DirPart);

f.s = regexpi({f.d.name},['mat',filesep,'SampleReport\w*.xls.mat'],'match');
f.s = [f.s{:}];

f.f = regexpi({f.d.name},['mat',filesep,'FixationReport\w*.xls.mat'],'match');
f.f = [f.f{:}];

%% Read data
try 
    load('data.m')
catch e
    
    data=struct();
    for i = 1:length(f.s)
       data.s{i}= struct2table(tdfread([f.DirPart,f.s{1}],'\t'));
    end
    for i = 1:length(f.f)
        data.f{i}= struct2table(tdfread([f.DirPart,f.f{1}],'\t'));
    end
   
    data.s=vertcat(data.s{:}); 
    data.f=vertcat(data.f{:});
    save('allData.mat','data','-v7.3')
end