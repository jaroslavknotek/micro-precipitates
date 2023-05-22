function refineMask()
InROI=false;
TheID=nan;
ImgName='D:\Git_Repos\TrainingData\Train_05-22\X\012_SMMAG_x300k_613';

Img=imread(fullfile(ImgName,'img.png'));
Mask=imread(fullfile(ImgName,'mask.png'));
Outlines=bwmorph(imfill(Mask),'remove');
Mask=bwlabel(Mask);
Outlines=Outlines.*Mask;

imagesc(Img,'ButtonDownFcn',@(s,e)imClick(s,e))
colormap('bone')

ROI=drawpolygon(gca,'Visible','off','Position',[1,1;2,2]);

Points=cell(1);
Outline(max(Mask,[],'all'))=line(nan,nan);
for ii=1:max(Mask,[],'all') % 186
    [y,x]=find(Outlines==ii);
    [xy]=sortPoints([x,y]);
%     close all
    xy=RDP(xy,.1);
    Outline(ii)=line([xy(:,1);xy(1,1)],[xy(:,2);xy(1,2)],'marker','.','color','r',...
        'buttondownfcn',@(s,e)lClick(s,e),'UserData',ii);
    Points{ii}=xy;

end



    function[sXY]=sortPoints(uXY)
        sXY=nan(size(uXY));
        CI=1;
        XY=uXY;
        for si=1:size(uXY,1)
            sXY(si,:)=XY(CI,:);
            XY=XY([1:CI-1,1+CI:end],:);
            EuD=sum([XY(:,1)-sXY(si,1),XY(:,2)-sXY(si,2)].^2,2);
            [~,CI]=min(EuD);
        end
    end

function[hull]=RDP(hull,eps)
        sp=hull(1,:);
        ep=hull(end,:);
        ip=hull(2:end-1,:);
        % calculate distances of inner points from the first-last line
        dst=PerpDist(ip,sp,ep);
        % find the point furthest from the f-l line
        [mx,mi]=max(dst);
        if mx>eps % furthest point does not fit in.
            lp=[sp;ip(1:mi,:)];
            if size(lp,1)>2
                lp=RDP(lp,eps);
            end
            rp=[ip(mi:end,:);ep];
            if size(rp,1)>2
                rp=RDP(rp,eps);
            end
            hull=[lp;rp(2:end,:)];
        else
            hull=[sp;ep];
        end
    end

    function[D]=PerpDist(PDX,varargin)
        D=nan(size(PDX,1),1);
        if nargin<2+1 % edge is defined by one point, use euclidean distance between points
            PDA=varargin{1};
            for PDi=1:size(PDX,1)
                D(PDi)=norm(PDX(PDi,:)-PDA);
            end
        else % edge is a line, use eucleidean distance from a line
            PDA=varargin{1};
            PDB=varargin{2};
            if PDA==PDB
                D=PerpDist(PDX,PDA);
            else
                for PDi=1:size(PDX,1)
                    D(PDi)=abs(PDA(1)*(PDB(2)-PDX(PDi,2))+PDB(1)*(PDX(PDi,2)-PDA(2))+PDX(PDi,1)*(PDA(2)-PDB(2)))/norm(PDB-PDA);
                end
            end
        end
    end

    function updateMask()
        Mask(Mask==TheID)=0;
        Layer=poly2mask(Points{TheID}(:,1),Points{TheID}(:,2),size(Mask,1),size(Mask,2))*TheID;
        set(Outline(TheID),'xdata',[Points{TheID}(:,1);Points{TheID}(1,1)],'ydata',[Points{TheID}(:,2);Points{TheID}(1,2)]);
        
        Mask=Mask+Layer;
        imwrite(Mask>0,fullfile(ImgName,'mask2.png'));
    end
    
    function imClick(~,e)
%         InROI,e
        if InROI
            if e.Button==3
                InROI=false;
                set(Outline,'visible','on')
                Points{TheID}=ROI.Position;
                ROI.Visible='off';
                updateMask();
            end
        else
            Click=floor(e.IntersectionPoint);
            X=Click(1);
            Y=Click(2);
    
            ID=Mask(Y,X);
            if ID~=0
                InROI=true;
                set(Outline,'visible','off');
                set(Outline(ID),'visible','on');
                TheID=ID;
                ROI.Visible='on';
%                 ID
                size(Points{ID})
                ROI.Position=Points{ID};
            else
                
            end
        end

    end
    function lClick(s,e)
        switch e.Button
            case 1
                InROI=true;
                TheID=s.UserData;
                ROI.Visible='on';
                ROI.Position=Points{TheID};
                set(Outline,'visible','off');
                set(Outline(TheID),'visible','off');
            case 3
                set(Outline(s.UserData),'xdata',nan,'ydata',nan);
                Mask(Mask==s.UserData)=0;
                Points{s.UserData}=nan(1,2);
        end
    end
end