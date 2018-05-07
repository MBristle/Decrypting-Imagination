


%%
xr= vertcat(ds(3).xr);
yr=vertcat(ds(3).yr);
a=[xr,yr];
%figure;
%plot(xr,yr,'*')
%hist3(a,'Ctrs',{0:25:1000 0:25:1000},'CDataMode','auto','FaceColor','interp')
%figure;

dsmean=[ds(:).mean];
dsstd=[ds(:).std];
mxr= nanmean([dsmean.xr]);
myr= nanmean([dsmean.yr]);
stdyr= nanmean([dsstd.yr]);
stdxr= nanmean([dsstd.xr]);
%%
t = tic;
f = waitbar(0,'starting...');
mat=cell(length(ds),1);
s=1200;
for ii= 1:length(ds)

    mat{ii}= zeros(s,s);
    xr= vertcat(ds(ii).xr);
    yr=vertcat(ds(ii).yr);
    a=[xr,yr];
    
    for i=1:length(a(:,1))
        
        tmp=gauss2d(s,s,(stdyr+stdxr)/length(a(:,1)),(stdyr+stdxr)/length(a(:,1)),a(i,2),a(i,1));
        mat{ii}= mat{ii}+tmp;
    end
    done=ii/length(ds);
    waitbar(done,f,...
            ['in progress: ',num2str(ii),'_',' ', '  estimated Time to finish: ', datestr(toc(t)/done*(1-done)/(24*60*60), 'DD:HH:MM:SS')]);

    
end

save('gaussMat.mat','mat','-v7.3')
%     surf(mat)
%     mesh(mat)
%     waitforbuttonpress
close(f)

%%
out=zeros(1000,1000);
for i=1:length(mat(:,1))
    mesh(mat{i})
    out=out+mat{i};
    waitforbuttonpress
end
%%
nCond= grp2idx({ds.condition});
out=cell(15,1);
for ii=1:15
    for iii=1:2
        idx=ii+15*(iii-1);
        a=find([ds.nImg]'==ii&nCond==iii);
        b=mat(a(:));
        out{idx}=zeros(1200,1200);
        for i=1:length(b(:,1))
            mesh(b{i})
            out{idx}=out{idx}+b{i};

        end
        mesh(out{idx})
        %waitforbuttonpress
    end
end
mesh(out{1})