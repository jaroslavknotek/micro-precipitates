if ~exist('Folder','var')
    Folder='C:';
end

[File,Folder,idx]=uigetfile(fullfile(Folder,'*.png'))

if idx==0
    clear all
    return
end
File=regexp(File,'\.','split');
File=File{1};


verifyMask(Folder,File)