set objFSO=CreateObject("scripting.filesystemobject")
set objSuperFolder=objFSO.GetFolder("C:\Users\Chochstr\Pictures\att_faces")
set obj=objFSO.CreateTextFile("C:\Users\Chochstr\Pictures\att_faces\Myfileslist.txt",8,True)
For Each subfolder in objSuperFolder.SubFolders
 Set objFolder = objFSO.GetFolder(subfolder.Path)
 Set colFiles = objFolder.Files
 For Each objFile in colFiles
  if UCase(objFSO.GetExtensionName(objFile.name)) = "PGM" or UCase(objFSO.GetExtensionName(objFile.name)) = "PNG" then
   s = Split(objFile.Path,"\")
   i = Mid(s(5),2)-1
   obj.Writeline(objFile.Path & ";" & i)
  End If
 Next	
Next
obj.Close
WScript.Echo "Done."
WScript.Quit 0