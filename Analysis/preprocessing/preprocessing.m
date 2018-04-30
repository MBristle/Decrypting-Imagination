%% preprocessing Fixation
%% construct path
clear
addpath('FuncSimToolbox','ScanMatch','LIBSVM')

f=struct();
f.parts = strsplit(pwd, filesep);
f.DirPart = fullfile( filesep,f.parts{1:end-2},['EyetrackingData',filesep,'mat',filesep]);
tmp=load([f.DirPart,'FixationReportAllData.xls.mat']);
data.raw=tmp.data;
clear('f','tmp')
% SAMPLE_INDEX ca.15552; samples/trail starts with 1 for each trail
% write sample analysis (long to wide format)



%% get metadata about the set
OUT = regexp(table2cell(data.raw(:,1)),...
    '^(?<participant>[a-zA-Z][a-zA-Z]\d\d)_(?<session>\d)$', 'names');
data.raw= [data.raw,struct2table([OUT{:}])];

data.md.trails=1:90;
data.md.sessions=unique(data.raw.session);
data.md.participants=unique(data.raw.participant);
clear('OUT')
%% Inspection
% SUMMARY: all participants: o_Landschaft_13.jpg is missing in block 4
% ug93_4: o_Landschaft_07.jpg in Block 5!
% 
% for p= 1:length(data.md.participants)
%     for s= 1:5
%         tmp=~cellfun(@isempty,regexp(table2cell(data.raw(:,1)),[data.md.participants{p},'_',num2str(s)], 'match'));
%         unique([num2str(data.raw.block(tmp,:)),data.raw.stim_name(tmp,:)],'rows')
%         disp(['name: ', data.md.participants(p),' session: ',s, ' max: ', max(data.raw.TRIAL_INDEX(tmp,:)),...
%         ' unique: ', unique(data.raw.TRIAL_INDEX(tmp,:)),...
%         ' pictures_block: ', length(unique([num2str(data.raw.block(tmp,:)),data.raw.stim_name(tmp,:)],'rows')), 'should be 90'])
%         
%     end
% end
% 
% % data.raw(data.raw.TRIAL_INDEX==91,:)
% tmp=data.raw(~cellfun(@isempty,regexp(table2cell(data.raw(:,1)),'ug93_4', 'match')),:);
% %plot(tmp.TRIAL_INDEX,tmp.TRIAL_FIXATION_TOTAL) % Trail 61 is missing and Trail 62 has only 1 Fixation
% histogram(categorical(cellstr(tmp.stim_name)))
% unique([num2str(tmp.block),tmp.stim_name],'rows') 

%% exclude trails for balanced design in crossvalidation -> see inspection for further detail
missing=false(height(data.raw),1);
missing(cellfun(@isempty,regexp(table2cell(data.raw(:,24)),'[KLG][a-z]*_\d\d?', 'match')))=true;
drop=~(cellfun(@str2double,table2cell(data.raw(:,21)))==4|... drop block 4 from all sessions and participants
    (cellfun(@str2double,table2cell(data.raw(:,21)))==5&...block 5
    ~cellfun(@isempty,regexp(table2cell(data.raw(:,1)),'ug93_4', 'match')))|missing); %participant 'ug93_4'
    

data.raw = data.raw(drop,:); % excludes trails
clear('drop','missing')


%% Setup Dataset
% ID: participant_session_block_stim_name

pic=regexp(table2cell(data.raw(:,24)),'[KLG][a-z]*_\d\d?', 'match');
datasetup = cellstr( unique([data.raw.RECORDING_SESSION_LABEL,...
    repmat('_',height(data.raw),1),num2str(data.raw.block),...
    repmat('_',height(data.raw),1),char([pic{:}]')],'rows'));
datasetup = horzcat(datasetup, regexp((datasetup(:,1)),...
    '(?<vpn>[a-z][a-z]\d\d)_(?<session>\d)_(?<block>\d)_(?<kat>[KLG][a-z]*)_(?<img>\d\d?)', 'names'));
assert(length(datasetup)==1860) % part 5 * session 5* block 5* img 15 - 15 (from part ug93_s4_b5)= 1860

%% extract features

%parameters of the RQA
param.delay = 1;
param.embed = 1;
param.rescale = 0;
param.metric = 'euclidian';
param.adjacency=[];
param.linelength = 2;
param.radius=64;

%setup dataset and progress doc
dataset=cell(2*length(datasetup),1);
f = waitbar(0,'Please wait...');
t = tic;

for i=1:length(datasetup)
    for ii =1:2 %perception and imagination
        %index for dataset
        idx=i+length(datasetup)*(ii-1);
        
        % selection of fixation corresponding to current VPN, Session,
        % Block, Picture and condition (perception or imagination)
         selection= ~cellfun(@isempty,...
             regexp(cellstr([data.raw.RECORDING_SESSION_LABEL,num2str(data.raw.block),data.raw{:,24}]),...
                [datasetup{i,2}.vpn,'_',datasetup{i,2}.session,datasetup{i,2}.block,...
                'o_',datasetup{i,2}.kat,'_',datasetup{i,2}.img,'.jpg'],...
                'match'))&...
             (ii==1&data.raw.CURRENT_FIX_START<15000)...perception$
             |(ii==2&data.raw.CURRENT_FIX_START>15000); %imagination
         
         %copies struct with cpn, session, block, kat, img information to
         %dataset
         
        % Sample description metadata
        dataset{idx}=datasetup{i,2};
        dataset{idx}.session=str2double(dataset{idx}.session);
        dataset{idx}.block=str2double(dataset{idx}.block);
        

        
        dataset{idx}.trailID=datasetup{i,1};
        
        if ii==1, dataset{idx}.condition = 'perception'; else dataset{idx}.condition='imagination'; end;

        % raw data
        dataset{idx}.xr=data.raw.CURRENT_FIX_X(selection);
        dataset{idx}.xl=str2double(cellstr(data.raw.CURRENT_FIX_X_OTHER(selection,:)));
        dataset{idx}.yr=data.raw.CURRENT_FIX_Y(selection);
        dataset{idx}.yl=str2double(cellstr(data.raw.CURRENT_FIX_X_OTHER(selection,:)));
        dataset{idx}.dur=data.raw.CURRENT_FIX_DURATION(selection);
        dataset{idx}.pupil=data.raw.CURRENT_FIX_PUPIL(selection);
        dataset{idx}.blink=data.raw.CURRENT_FIX_BLINK_AROUND(selection,:);
       
                
        %stage 1 
        dataset{idx}.numberOfBlink=sum(~cellfun(@isempty,regexp(cellstr(dataset{idx}.blink),'AFTER', 'match'))); %
        dataset{idx}.numberOfFix=length(dataset{idx}.xr); % check typ
        dataset{idx}.rating=str2double(unique(data.raw.Rating(selection))); % check for typ
        
        dataset{idx}.mean.xr=mean(dataset{idx}.xr);
        dataset{idx}.mean.yr=nanmean(dataset{idx}.xl);
        dataset{idx}.mean.xl=mean(dataset{idx}.yr);
        dataset{idx}.mean.yl=nanmean(dataset{idx}.yl);
        dataset{idx}.mean.dur=mean(dataset{idx}.dur);
        dataset{idx}.mean.pupil=mean(dataset{idx}.pupil);
        
        dataset{idx}.std.xr=std(dataset{idx}.xr);
        dataset{idx}.std.yr=nanstd(dataset{idx}.xl);
        dataset{idx}.std.xl=std(dataset{idx}.yr);
        dataset{idx}.std.yl=nanstd(dataset{idx}.yl);
        dataset{idx}.std.dur=std(dataset{idx}.dur);
        dataset{idx}.std.pupil=std(dataset{idx}.pupil);
        
        %stage 2 
        
        %todo: probability map of X,Y to be fixated --> bootstrap
        %distribution? 
        %todo: PDF of duration 
        %todo: PDF of pupil

        
        %stage 3
        %dataset{idx}.rqa=Rqa([dataset{idx}.xr,dataset{idx}.yr],param);    
        %dataset{idx}.scanMatch= %implement scanmatch
        %dataset{idx}.multiMatch= %implement multiMatch
        
        %stage 4
        
        %progress Doc
        done=sum(~cellfun(@isempty,dataset))/length(dataset);
        waitbar(done,f,...
            ['in progress: ',strrep(dataset{idx}.trailID,'_',' '), '  estimated Time to finish: ', datestr(toc(t)/done*(1-done)/(24*60*60), 'DD:HH:MM:SS')]);

    end
end
waitbar(1,f,'saving File: dataset.mat');
%% numeric VPN, Img, Category
ds= [dataset{:}];
nCat= grp2idx({ds.kat}'); 
nImg= grp2idx(horzcat(num2str(grp2idx({ds.kat}')),vertcat(ds.img)));
nVpn= grp2idx({ds.vpn});


for i=1:length(ds)
    ds(i).nCat=nCat(i);
    ds(i).nImg=nImg(i);
    ds(i).nVpn=nVpn(i);
end

save('dataset_raw_sum.mat','ds','-v7.3') %save to HDF5 file
close(f)
