// CkScp.h: interface for the CkScp class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkScp_H
#define _CkScp_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacks.h"

class CkTask;
class CkBinData;
class CkByteData;
class CkSsh;
class CkBaseProgress;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkScp
class CK_VISIBLE_PUBLIC CkScp  : public CkClassWithCallbacks
{
    private:

	// Don't allow assignment or copying these objects.
	CkScp(const CkScp &);
	CkScp &operator=(const CkScp &);

    public:
	CkScp(void);
	virtual ~CkScp(void);

	static CkScp *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	CkBaseProgress *get_EventCallbackObject(void) const;
	void put_EventCallbackObject(CkBaseProgress *progress);


	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// When set to true, causes the currently running method to abort. Methods that
	// always finish quickly (i.e.have no length file operations or network
	// communications) are not affected. If no method is running, then this property is
	// automatically reset to false when the next method is called. When the abort
	// occurs, this property is reset to false. Both synchronous and asynchronous
	// method calls can be aborted. (A synchronous method call could be aborted by
	// setting this property from a separate thread.)
	bool get_AbortCurrent(void);
	// When set to true, causes the currently running method to abort. Methods that
	// always finish quickly (i.e.have no length file operations or network
	// communications) are not affected. If no method is running, then this property is
	// automatically reset to false when the next method is called. When the abort
	// occurs, this property is reset to false. Both synchronous and asynchronous
	// method calls can be aborted. (A synchronous method call could be aborted by
	// setting this property from a separate thread.)
	void put_AbortCurrent(bool newVal);

	// This is the number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any SSH operation prior to
	// completion. If HeartbeatMs is 0 (the default), no AbortCheck event callbacks
	// will fire.
	int get_HeartbeatMs(void);
	// This is the number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any SSH operation prior to
	// completion. If HeartbeatMs is 0 (the default), no AbortCheck event callbacks
	// will fire.
	void put_HeartbeatMs(int newVal);

	// This property is only valid in programming environment and languages that allow
	// for event callbacks.
	// 
	// Sets the value to be defined as 100% complete for the purpose of PercentDone
	// event callbacks. The defaut value of 100 means that at most 100 event
	// PercentDone callbacks will occur in a method that (1) is event enabled and (2)
	// is such that it is possible to measure progress as a percentage completed. This
	// property may be set to larger numbers to get more fine-grained PercentDone
	// callbacks. For example, setting this property equal to 1000 will provide
	// callbacks with .1 percent granularity. For example, a value of 453 would
	// indicate 45.3% competed. This property is clamped to a minimum value of 10, and
	// a maximum value of 100000.
	// 
	int get_PercentDoneScale(void);
	// This property is only valid in programming environment and languages that allow
	// for event callbacks.
	// 
	// Sets the value to be defined as 100% complete for the purpose of PercentDone
	// event callbacks. The defaut value of 100 means that at most 100 event
	// PercentDone callbacks will occur in a method that (1) is event enabled and (2)
	// is such that it is possible to measure progress as a percentage completed. This
	// property may be set to larger numbers to get more fine-grained PercentDone
	// callbacks. For example, setting this property equal to 1000 will provide
	// callbacks with .1 percent granularity. For example, a value of 453 would
	// indicate 45.3% competed. This property is clamped to a minimum value of 10, and
	// a maximum value of 100000.
	// 
	void put_PercentDoneScale(int newVal);

	// A JSON string specifying environment variables that are to be set for each SCP
	// upload or download. For example:
	// {
	//     "LCS_PASSWORD": "myPassword",
	//     "SOME_ENV_VAR": "some_value",
	// ...
	// }
	void get_SendEnv(CkString &str);
	// A JSON string specifying environment variables that are to be set for each SCP
	// upload or download. For example:
	// {
	//     "LCS_PASSWORD": "myPassword",
	//     "SOME_ENV_VAR": "some_value",
	// ...
	// }
	const char *sendEnv(void);
	// A JSON string specifying environment variables that are to be set for each SCP
	// upload or download. For example:
	// {
	//     "LCS_PASSWORD": "myPassword",
	//     "SOME_ENV_VAR": "some_value",
	// ...
	// }
	void put_SendEnv(const char *newVal);

	// The paths of the files uploaded or downloaded in the last call to SyncUploadTree
	// or SyncDownloadTree. The paths are listed one per line. In both cases (for
	// upload and download) each line contains the full local file path (not the remote
	// path).
	void get_SyncedFiles(CkString &str);
	// The paths of the files uploaded or downloaded in the last call to SyncUploadTree
	// or SyncDownloadTree. The paths are listed one per line. In both cases (for
	// upload and download) each line contains the full local file path (not the remote
	// path).
	const char *syncedFiles(void);
	// The paths of the files uploaded or downloaded in the last call to SyncUploadTree
	// or SyncDownloadTree. The paths are listed one per line. In both cases (for
	// upload and download) each line contains the full local file path (not the remote
	// path).
	void put_SyncedFiles(const char *newVal);

	// Can contain a wildcarded list of filename patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will only transfer files having a filename that matches any one of these
	// patterns.
	void get_SyncMustMatch(CkString &str);
	// Can contain a wildcarded list of filename patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will only transfer files having a filename that matches any one of these
	// patterns.
	const char *syncMustMatch(void);
	// Can contain a wildcarded list of filename patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will only transfer files having a filename that matches any one of these
	// patterns.
	void put_SyncMustMatch(const char *newVal);

	// Can contain a wildcarded list of directory name patterns separated by
	// semicolons. For example, "a*; b*; c*". If set, the SyncTreeUpload and
	// SyncTreeDownload methods will only traverse into a directory that matches any
	// one of these patterns.
	void get_SyncMustMatchDir(CkString &str);
	// Can contain a wildcarded list of directory name patterns separated by
	// semicolons. For example, "a*; b*; c*". If set, the SyncTreeUpload and
	// SyncTreeDownload methods will only traverse into a directory that matches any
	// one of these patterns.
	const char *syncMustMatchDir(void);
	// Can contain a wildcarded list of directory name patterns separated by
	// semicolons. For example, "a*; b*; c*". If set, the SyncTreeUpload and
	// SyncTreeDownload methods will only traverse into a directory that matches any
	// one of these patterns.
	void put_SyncMustMatchDir(const char *newVal);

	// Can contain a wildcarded list of filename patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will not transfer files having a filename that matches any one of these
	// patterns.
	void get_SyncMustNotMatch(CkString &str);
	// Can contain a wildcarded list of filename patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will not transfer files having a filename that matches any one of these
	// patterns.
	const char *syncMustNotMatch(void);
	// Can contain a wildcarded list of filename patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will not transfer files having a filename that matches any one of these
	// patterns.
	void put_SyncMustNotMatch(const char *newVal);

	// Can contain a wildcarded list of directory name patterns separated by
	// semicolons. For example, "a*; b*; c*". If set, the SyncTreeUpload and
	// SyncTreeDownload methods will not traverse into a directory that matches any one
	// of these patterns.
	void get_SyncMustNotMatchDir(CkString &str);
	// Can contain a wildcarded list of directory name patterns separated by
	// semicolons. For example, "a*; b*; c*". If set, the SyncTreeUpload and
	// SyncTreeDownload methods will not traverse into a directory that matches any one
	// of these patterns.
	const char *syncMustNotMatchDir(void);
	// Can contain a wildcarded list of directory name patterns separated by
	// semicolons. For example, "a*; b*; c*". If set, the SyncTreeUpload and
	// SyncTreeDownload methods will not traverse into a directory that matches any one
	// of these patterns.
	void put_SyncMustNotMatchDir(const char *newVal);

	// This is a catch-all property to be used for uncommon needs. The default value is
	// the empty string.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     FilenameOnly - Introduced in v9.5.0.77. Set this property to the keyword
	//     "FilenameOnly" if only the filename should be used in the "scp -t" command.
	//     (LANCOM routers using SCP seem to need it.)
	//     ProtectFromVpn - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	// 
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. The default value is
	// the empty string.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     FilenameOnly - Introduced in v9.5.0.77. Set this property to the keyword
	//     "FilenameOnly" if only the filename should be used in the "scp -t" command.
	//     (LANCOM routers using SCP seem to need it.)
	//     ProtectFromVpn - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	// 
	const char *uncommonOptions(void);
	// This is a catch-all property to be used for uncommon needs. The default value is
	// the empty string.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     FilenameOnly - Introduced in v9.5.0.77. Set this property to the keyword
	//     "FilenameOnly" if only the filename should be used in the "scp -t" command.
	//     (LANCOM routers using SCP seem to need it.)
	//     ProtectFromVpn - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	// 
	void put_UncommonOptions(const char *newVal);

	// When Chilkat uploads a file by SCP, the UNIX permissions of the remote file are
	// set based on the permissions of the local file being uploaded. Usually this is
	// OK, but in some cases the access permissions of the local file are not what is
	// wanted for the remote file. This property can be set to an octal permissions
	// string, such as "0644", to force the remote file permissions to this value.
	// 
	// The default value of this property is the empty string (remote files permissions
	// mirror the permissions of the local file being uploaded).
	// 
	void get_UnixPermOverride(CkString &str);
	// When Chilkat uploads a file by SCP, the UNIX permissions of the remote file are
	// set based on the permissions of the local file being uploaded. Usually this is
	// OK, but in some cases the access permissions of the local file are not what is
	// wanted for the remote file. This property can be set to an octal permissions
	// string, such as "0644", to force the remote file permissions to this value.
	// 
	// The default value of this property is the empty string (remote files permissions
	// mirror the permissions of the local file being uploaded).
	// 
	const char *unixPermOverride(void);
	// When Chilkat uploads a file by SCP, the UNIX permissions of the remote file are
	// set based on the permissions of the local file being uploaded. Usually this is
	// OK, but in some cases the access permissions of the local file are not what is
	// wanted for the remote file. This property can be set to an octal permissions
	// string, such as "0644", to force the remote file permissions to this value.
	// 
	// The default value of this property is the empty string (remote files permissions
	// mirror the permissions of the local file being uploaded).
	// 
	void put_UnixPermOverride(const char *newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Downloads a binary file from the SSH server and appends to the contents of bd.
	bool DownloadBd(const char *remotePath, CkBinData &bd);

	// Downloads a binary file from the SSH server and appends to the contents of bd.
	CkTask *DownloadBdAsync(const char *remotePath, CkBinData &bd);


	// Downloads a binary file from the SSH server and returns the contents.
	bool DownloadBinary(const char *remotePath, CkByteData &outBytes);

	// Downloads a binary file from the SSH server and returns the contents.
	CkTask *DownloadBinaryAsync(const char *remotePath);


	// Downloads a file from the SSH server returns the contents in an encoded string
	// (using an encoding such as base64).
	bool DownloadBinaryEncoded(const char *remotePath, const char *encoding, CkString &outStr);

	// Downloads a file from the SSH server returns the contents in an encoded string
	// (using an encoding such as base64).
	const char *downloadBinaryEncoded(const char *remotePath, const char *encoding);
	// Downloads a file from the SSH server returns the contents in an encoded string
	// (using an encoding such as base64).
	CkTask *DownloadBinaryEncodedAsync(const char *remotePath, const char *encoding);


	// Downloads a file from the remote SSH server to the local filesystem.
	bool DownloadFile(const char *remotePath, const char *localPath);

	// Downloads a file from the remote SSH server to the local filesystem.
	CkTask *DownloadFileAsync(const char *remotePath, const char *localPath);


	// Downloads a text file from the SSH server and returns the contents as a string.
	bool DownloadString(const char *remotePath, const char *charset, CkString &outStr);

	// Downloads a text file from the SSH server and returns the contents as a string.
	const char *downloadString(const char *remotePath, const char *charset);
	// Downloads a text file from the SSH server and returns the contents as a string.
	CkTask *DownloadStringAsync(const char *remotePath, const char *charset);


	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTask &task);


	// Downloads files from the SSH server to a local directory tree. Synchronization
	// modes include:
	// 
	//     mode=0: Download all files
	//     mode=1: Download all files that do not exist on the local filesystem.
	//     mode=2: Download newer or non-existant files.
	//     mode=3: Download only newer files. If a file does not already exist on the
	//     local filesystem, it is not downloaded from the server.
	//     mode=5: Download only missing files or files with size differences.
	//     mode=6: Same as mode 5, but also download newer files.
	//     
	// 
	bool SyncTreeDownload(const char *remoteRoot, const char *localRoot, int mode, bool bRecurse);

	// Downloads files from the SSH server to a local directory tree. Synchronization
	// modes include:
	// 
	//     mode=0: Download all files
	//     mode=1: Download all files that do not exist on the local filesystem.
	//     mode=2: Download newer or non-existant files.
	//     mode=3: Download only newer files. If a file does not already exist on the
	//     local filesystem, it is not downloaded from the server.
	//     mode=5: Download only missing files or files with size differences.
	//     mode=6: Same as mode 5, but also download newer files.
	//     
	// 
	CkTask *SyncTreeDownloadAsync(const char *remoteRoot, const char *localRoot, int mode, bool bRecurse);


	// Uploads a directory tree from the local filesystem to the SSH server.
	// Synchronization modes include:
	// 
	//     mode=0: Upload all files
	//     mode=1: Upload all files that do not exist on the FTP server.
	//     mode=2: Upload newer or non-existant files.
	//     mode=3: Upload only newer files. If a file does not already exist on the FTP
	//     server, it is not uploaded.
	//     mode=4: transfer missing files or files with size differences.
	//     mode=5: same as mode 4, but also newer files.
	// 
	bool SyncTreeUpload(const char *localBaseDir, const char *remoteBaseDir, int mode, bool bRecurse);

	// Uploads a directory tree from the local filesystem to the SSH server.
	// Synchronization modes include:
	// 
	//     mode=0: Upload all files
	//     mode=1: Upload all files that do not exist on the FTP server.
	//     mode=2: Upload newer or non-existant files.
	//     mode=3: Upload only newer files. If a file does not already exist on the FTP
	//     server, it is not uploaded.
	//     mode=4: transfer missing files or files with size differences.
	//     mode=5: same as mode 4, but also newer files.
	// 
	CkTask *SyncTreeUploadAsync(const char *localBaseDir, const char *remoteBaseDir, int mode, bool bRecurse);


	// Uploads the contents of bd to a file on the SSH server.
	bool UploadBd(const char *remotePath, CkBinData &bd);

	// Uploads the contents of bd to a file on the SSH server.
	CkTask *UploadBdAsync(const char *remotePath, CkBinData &bd);


	// Uploads binary data to a file on the SSH server.
	bool UploadBinary(const char *remotePath, CkByteData &binData);

	// Uploads binary data to a file on the SSH server.
	CkTask *UploadBinaryAsync(const char *remotePath, CkByteData &binData);


	// Uploads the binary data to a file on the remote SSH server. The binary data is
	// passed in encoded string representation (such as base64, or hex).
	bool UploadBinaryEncoded(const char *remotePath, const char *encodedData, const char *encoding);

	// Uploads the binary data to a file on the remote SSH server. The binary data is
	// passed in encoded string representation (such as base64, or hex).
	CkTask *UploadBinaryEncodedAsync(const char *remotePath, const char *encodedData, const char *encoding);


	// Uploads a file from the local filesystem to the remote SSH server.
	bool UploadFile(const char *localPath, const char *remotePath);

	// Uploads a file from the local filesystem to the remote SSH server.
	CkTask *UploadFileAsync(const char *localPath, const char *remotePath);


	// Uploads the contents of a string to a file on the remote SSH server.
	bool UploadString(const char *remotePath, const char *textData, const char *charset);

	// Uploads the contents of a string to a file on the remote SSH server.
	CkTask *UploadStringAsync(const char *remotePath, const char *textData, const char *charset);


	// Uses the SSH connection of sshConnection for the SCP transfers. All of the connection and
	// socket related properties, proxy properites, timeout properties, session log,
	// etc. set on the SSH object apply to the SCP methods (because internally it is
	// the SSH object that is used to do the work of the file transfers).
	// 
	// Note: There is no UnlockComponent method in the SCP class because it is the SSH
	// object that must be unlocked. When the SSH object is not unlocked, this method
	// will return false to indicate failure.
	// 
	bool UseSsh(CkSsh &sshConnection);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
