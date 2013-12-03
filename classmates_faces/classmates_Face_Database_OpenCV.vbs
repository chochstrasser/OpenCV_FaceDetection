set objFSO=CreateObject("scripting.filesystemobject")
set objSuperFolder=objFSO.GetFolder("C:\Users\Chochstr\Pictures\classmates_faces")
set obj=objFSO.CreateTextFile("C:\Users\Chochstr\Pictures\classmates_faces\Myfileslist.txt",8,True)
Dim oStream
Set oStream = CreateObject("ADODB.Stream")
oStream.CharSet = "utf-8"
oStream.Open
i = 0
For Each subfolder in objSuperFolder.SubFolders
 Set objFolder = objFSO.GetFolder(subfolder.Path)
 Set colFiles = objFolder.Files
 i = i + 1
 For Each objFile in colFiles
  if UCase(objFSO.GetExtensionName(objFile.name)) = "PNG" then
   s = Split(objFile.Path,"\")
   oStream.WriteText(objFile.Path & ";" & i & vbNewLine)
   'obj.Writeline(objFile.Path & ";" & i)
  End If
 Next	
Next
obj.Close
Ostream.SaveToFile "C:\Users\Chochstr\Pictures\classmates_faces\Myfileslist.txt", 2
oStream.Close
WScript.Echo "Done."
WScript.Quit 0