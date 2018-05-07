%% construct path
clear
%addpath('FuncSimToolbox','ScanMatch','LIBSVM')

f=struct();
f.parts = strsplit(pwd, '/');
f.DirPart = fullfile( '/',f.parts{1:end-2},'EyetrackingData/mat/');
f.d =dir(f.DirPart);

f.s = regexpi({f.d.name},'SampleReport\w*.xls.mat','match');
f.s = [f.s{:}];

f.f = regexpi({f.d.name},'FixationReport\w*.xls.mat','match');
f.f = [f.f{:}];

%% Read data

data=struct();
for i = 1:length(f.s)
   tmp=load([f.DirPart,f.s{i}]);
   if i==1
        data.s_head= tmp.data.s.Properties.VariableNames;
   end
   data.s{i}=table2cell(tmp.data.s);
end

for i = 1:length(f.f)
   tmp=load([f.DirPart,f.f{i}]);
   data.f{i}=tmp.data.f ;
end

data.s=vertcat(data.s{:}); 
data.f=vertcat(data.f{:});
save('allData.mat','data','-v7.3')
