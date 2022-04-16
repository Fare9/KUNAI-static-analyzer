// CkTarW.h: interface for the CkTarW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkTarW_H
#define _CkTarW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacksW.h"

class CkTaskW;
class CkBinDataW;
class CkByteData;
class CkTarProgressW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkTarW
class CK_VISIBLE_PUBLIC CkTarW  : public CkClassWithCallbacksW
{
	private:
	bool m_cbOwned;

	private:
	
	// Don't allow assignment or copying these objects.
	CkTarW(const CkTarW &);
	CkTarW &operator=(const CkTarW &);

    public:
	CkTarW(void);
	virtual ~CkTarW(void);

	

	static CkTarW *createNew(void);
	

	CkTarW(bool bCallbackOwned);
	static CkTarW *createNew(bool bCallbackOwned);

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	CkTarProgressW *get_EventCallbackObject(void) const;
	void put_EventCallbackObject(CkTarProgressW *progress);


	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// If true, then untar methods, such as Untar, UntarGz, UntarBz2, and UntarZ,
	// will also capture an XML listing of the contents in the XmlListing property. The
	// format of the XML contained in XmlListing is identical to what is returned by
	// the ListXml method. The default value is false.
	bool get_CaptureXmlListing(void);
	// If true, then untar methods, such as Untar, UntarGz, UntarBz2, and UntarZ,
	// will also capture an XML listing of the contents in the XmlListing property. The
	// format of the XML contained in XmlListing is identical to what is returned by
	// the ListXml method. The default value is false.
	void put_CaptureXmlListing(bool newVal);

	// Character encoding to be used when interpreting filenames within .tar archives
	// for untar operations. The default is "utf-8", and this is typically not changed.
	// (The WriteTar methods always uses utf-8 to store filenames within the TAR
	// archive.)
	void get_Charset(CkString &str);
	// Character encoding to be used when interpreting filenames within .tar archives
	// for untar operations. The default is "utf-8", and this is typically not changed.
	// (The WriteTar methods always uses utf-8 to store filenames within the TAR
	// archive.)
	const wchar_t *charset(void);
	// Character encoding to be used when interpreting filenames within .tar archives
	// for untar operations. The default is "utf-8", and this is typically not changed.
	// (The WriteTar methods always uses utf-8 to store filenames within the TAR
	// archive.)
	void put_Charset(const wchar_t *newVal);

	// The directory permissions to used in WriteTar* methods. The default is octal
	// 0755. This is the value to be stored in the "mode" field of each TAR header for
	// a directory entries.
	int get_DirMode(void);
	// The directory permissions to used in WriteTar* methods. The default is octal
	// 0755. This is the value to be stored in the "mode" field of each TAR header for
	// a directory entries.
	void put_DirMode(int newVal);

	// A prefix to be added to each file's path within the TAR archive as it is being
	// created. For example, if this property is set to the string "subdir1", then
	// "subdir1/" will be prepended to each file's path within the TAR.
	// 
	// Note: This property does not apply to files added using the AddFile2 method,
	// which directly specifies the path-in-tar.
	// 
	void get_DirPrefix(CkString &str);
	// A prefix to be added to each file's path within the TAR archive as it is being
	// created. For example, if this property is set to the string "subdir1", then
	// "subdir1/" will be prepended to each file's path within the TAR.
	// 
	// Note: This property does not apply to files added using the AddFile2 method,
	// which directly specifies the path-in-tar.
	// 
	const wchar_t *dirPrefix(void);
	// A prefix to be added to each file's path within the TAR archive as it is being
	// created. For example, if this property is set to the string "subdir1", then
	// "subdir1/" will be prepended to each file's path within the TAR.
	// 
	// Note: This property does not apply to files added using the AddFile2 method,
	// which directly specifies the path-in-tar.
	// 
	void put_DirPrefix(const wchar_t *newVal);

	// The file permissions to used in WriteTar* methods. The default is octal 0644.
	// This is the value to be stored in the "mode" field of each TAR header for a file
	// entries.
	int get_FileMode(void);
	// The file permissions to used in WriteTar* methods. The default is octal 0644.
	// This is the value to be stored in the "mode" field of each TAR header for a file
	// entries.
	void put_FileMode(int newVal);

	// The default numerical GID to be stored in each TAR header when writing TAR
	// archives. The default value is 1000.
	int get_GroupId(void);
	// The default numerical GID to be stored in each TAR header when writing TAR
	// archives. The default value is 1000.
	void put_GroupId(int newVal);

	// The default group name to be stored in each TAR header when writing TAR
	// archives. The default value is the logged-on username of the application's
	// process.
	void get_GroupName(CkString &str);
	// The default group name to be stored in each TAR header when writing TAR
	// archives. The default value is the logged-on username of the application's
	// process.
	const wchar_t *groupName(void);
	// The default group name to be stored in each TAR header when writing TAR
	// archives. The default value is the logged-on username of the application's
	// process.
	void put_GroupName(const wchar_t *newVal);

	// This is the number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any TAR operation prior to
	// completion. If HeartbeatMs is 0, no AbortCheck event callbacks will occur.
	int get_HeartbeatMs(void);
	// This is the number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any TAR operation prior to
	// completion. If HeartbeatMs is 0, no AbortCheck event callbacks will occur.
	void put_HeartbeatMs(int newVal);

	// Determines whether pattern matching for the MustMatch and MustNotMatch
	// properties is case-sensitive or not. The default value is false.
	bool get_MatchCaseSensitive(void);
	// Determines whether pattern matching for the MustMatch and MustNotMatch
	// properties is case-sensitive or not. The default value is false.
	void put_MatchCaseSensitive(bool newVal);

	// If set, then file paths must match this pattern when creating TAR archives, or
	// when extracting from TAR archives. If a file does not match, it will not be
	// included when creating a TAR, or it will not be extracted when extracting from a
	// TAR. This property also applies to methods that create or extract from
	// compressed TAR archives.
	// 
	// The must-match pattern may include 0 or more asterisk characters, each of which
	// represents 0 or more of any character. For example, the pattern "*.txt" causes
	// only .txt files to be included or extracted. The default value is an empty
	// string, indicating that all files are implicitly matched.
	// 
	void get_MustMatch(CkString &str);
	// If set, then file paths must match this pattern when creating TAR archives, or
	// when extracting from TAR archives. If a file does not match, it will not be
	// included when creating a TAR, or it will not be extracted when extracting from a
	// TAR. This property also applies to methods that create or extract from
	// compressed TAR archives.
	// 
	// The must-match pattern may include 0 or more asterisk characters, each of which
	// represents 0 or more of any character. For example, the pattern "*.txt" causes
	// only .txt files to be included or extracted. The default value is an empty
	// string, indicating that all files are implicitly matched.
	// 
	const wchar_t *mustMatch(void);
	// If set, then file paths must match this pattern when creating TAR archives, or
	// when extracting from TAR archives. If a file does not match, it will not be
	// included when creating a TAR, or it will not be extracted when extracting from a
	// TAR. This property also applies to methods that create or extract from
	// compressed TAR archives.
	// 
	// The must-match pattern may include 0 or more asterisk characters, each of which
	// represents 0 or more of any character. For example, the pattern "*.txt" causes
	// only .txt files to be included or extracted. The default value is an empty
	// string, indicating that all files are implicitly matched.
	// 
	void put_MustMatch(const wchar_t *newVal);

	// If set, then file paths must NOT match this pattern when creating TAR archives,
	// or when extracting from TAR archives. If a file path matches, it will not be
	// included when creating a TAR, or it will not be extracted when extracting from a
	// TAR. This property also applies to methods that create or extract from
	// compressed TAR archives.
	// 
	// The must-not-match pattern may include 0 or more asterisk characters, each of
	// which represents 0 or more of any character. For example, the pattern "*.obj"
	// causes all .obj files to be skipped. The default value is an empty string,
	// indicating that no files are skipped.
	// 
	void get_MustNotMatch(CkString &str);
	// If set, then file paths must NOT match this pattern when creating TAR archives,
	// or when extracting from TAR archives. If a file path matches, it will not be
	// included when creating a TAR, or it will not be extracted when extracting from a
	// TAR. This property also applies to methods that create or extract from
	// compressed TAR archives.
	// 
	// The must-not-match pattern may include 0 or more asterisk characters, each of
	// which represents 0 or more of any character. For example, the pattern "*.obj"
	// causes all .obj files to be skipped. The default value is an empty string,
	// indicating that no files are skipped.
	// 
	const wchar_t *mustNotMatch(void);
	// If set, then file paths must NOT match this pattern when creating TAR archives,
	// or when extracting from TAR archives. If a file path matches, it will not be
	// included when creating a TAR, or it will not be extracted when extracting from a
	// TAR. This property also applies to methods that create or extract from
	// compressed TAR archives.
	// 
	// The must-not-match pattern may include 0 or more asterisk characters, each of
	// which represents 0 or more of any character. For example, the pattern "*.obj"
	// causes all .obj files to be skipped. The default value is an empty string,
	// indicating that no files are skipped.
	// 
	void put_MustNotMatch(const wchar_t *newVal);

	// If true, then absolute paths are converted to relative paths by removing the
	// leading "/" or "\" character when untarring. This protects your system from
	// unknowingly untarring files into important system directories, such as
	// C:\Windows\system32. The default value is true.
	bool get_NoAbsolutePaths(void);
	// If true, then absolute paths are converted to relative paths by removing the
	// leading "/" or "\" character when untarring. This protects your system from
	// unknowingly untarring files into important system directories, such as
	// C:\Windows\system32. The default value is true.
	void put_NoAbsolutePaths(bool newVal);

	// The total number of directory roots set by calling AddDirRoot (i.e. the number
	// of times AddDirRoot was called by the application). A TAR archive is created by
	// calling AddDirRoot for one or more directory tree roots, followed by a single
	// call to WriteTar (or WriteTarBz2, WriteTarGz, WriteTarZ). This allows for TAR
	// archives containing multiple directory trees to be created.
	int get_NumDirRoots(void);

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

	// The file permissions to used in WriteTar* methods for shell script files (.sh,
	// .csh, .bash, .bsh). The default is octal 0755. This is the value to be stored in
	// the "mode" field of each TAR header for a file entries.
	int get_ScriptFileMode(void);
	// The file permissions to used in WriteTar* methods for shell script files (.sh,
	// .csh, .bash, .bsh). The default is octal 0755. This is the value to be stored in
	// the "mode" field of each TAR header for a file entries.
	void put_ScriptFileMode(int newVal);

	// If true, then untar methods, such as Untar, UntarGz, UntarBz2, and UntarZ, do
	// not produce any output. Setting this value equal to true is useful when the
	// CaptureXmlListing is also set to true, which enables an application to get the
	// contents of a TAR archive without extracting. The default value is false.
	bool get_SuppressOutput(void);
	// If true, then untar methods, such as Untar, UntarGz, UntarBz2, and UntarZ, do
	// not produce any output. Setting this value equal to true is useful when the
	// CaptureXmlListing is also set to true, which enables an application to get the
	// contents of a TAR archive without extracting. The default value is false.
	void put_SuppressOutput(bool newVal);

	// This property is deprecated. Applications should instead use the
	// MatchCaseSensitive property. Until this property is officially removed, it will
	// behave the same as the MatchCaseSensitive property.
	bool get_UntarCaseSensitive(void);
	// This property is deprecated. Applications should instead use the
	// MatchCaseSensitive property. Until this property is officially removed, it will
	// behave the same as the MatchCaseSensitive property.
	void put_UntarCaseSensitive(bool newVal);

	// Similar to the VerboseLogging property. If set to true, then information about
	// each file/directory extracted in an untar method call is logged to LastErrorText
	// (or LastErrorXml / LastErrorHtml). The default value is false.
	bool get_UntarDebugLog(void);
	// Similar to the VerboseLogging property. If set to true, then information about
	// each file/directory extracted in an untar method call is logged to LastErrorText
	// (or LastErrorXml / LastErrorHtml). The default value is false.
	void put_UntarDebugLog(bool newVal);

	// If true, then discard all path information when untarring. This causes all
	// files to be untarred into a single directory. The default value is false.
	bool get_UntarDiscardPaths(void);
	// If true, then discard all path information when untarring. This causes all
	// files to be untarred into a single directory. The default value is false.
	void put_UntarDiscardPaths(bool newVal);

	// The directory path where files are extracted when untarring. The default value
	// is ".", meaning that the current working directory of the calling process is
	// used. If UntarDiscardPaths is set, then all files are untarred into this
	// directory. Otherwise, the untar operation will re-create a directory tree rooted
	// in this directory.
	void get_UntarFromDir(CkString &str);
	// The directory path where files are extracted when untarring. The default value
	// is ".", meaning that the current working directory of the calling process is
	// used. If UntarDiscardPaths is set, then all files are untarred into this
	// directory. Otherwise, the untar operation will re-create a directory tree rooted
	// in this directory.
	const wchar_t *untarFromDir(void);
	// The directory path where files are extracted when untarring. The default value
	// is ".", meaning that the current working directory of the calling process is
	// used. If UntarDiscardPaths is set, then all files are untarred into this
	// directory. Otherwise, the untar operation will re-create a directory tree rooted
	// in this directory.
	void put_UntarFromDir(const wchar_t *newVal);

	// This property is deprecated. The MustMatch property should be used instead.
	// Until this property is officially removed from the API, it is identical in
	// behavior the MustMatch property.
	void get_UntarMatchPattern(CkString &str);
	// This property is deprecated. The MustMatch property should be used instead.
	// Until this property is officially removed from the API, it is identical in
	// behavior the MustMatch property.
	const wchar_t *untarMatchPattern(void);
	// This property is deprecated. The MustMatch property should be used instead.
	// Until this property is officially removed from the API, it is identical in
	// behavior the MustMatch property.
	void put_UntarMatchPattern(const wchar_t *newVal);

	// Limits the number of files extracted during an untar to this count. The default
	// value is 0 to indicate no maximum. To untar a single file, one might set the
	// UntarMatchPattern such that it will match only the file to be extracted, and
	// also set UntarMaxCount equal to 1. This causes an untar to scan the TAR archive
	// until it finds the matching file, extract it, and then return.
	int get_UntarMaxCount(void);
	// Limits the number of files extracted during an untar to this count. The default
	// value is 0 to indicate no maximum. To untar a single file, one might set the
	// UntarMatchPattern such that it will match only the file to be extracted, and
	// also set UntarMaxCount equal to 1. This causes an untar to scan the TAR archive
	// until it finds the matching file, extract it, and then return.
	void put_UntarMaxCount(int newVal);

	// The default numerical UID to be stored in each TAR header when writing TAR
	// archives. The default value is 1000.
	int get_UserId(void);
	// The default numerical UID to be stored in each TAR header when writing TAR
	// archives. The default value is 1000.
	void put_UserId(int newVal);

	// The default user name to be stored in each TAR header when writing TAR archives.
	// The default value is the logged-on username of the application's process.
	void get_UserName(CkString &str);
	// The default user name to be stored in each TAR header when writing TAR archives.
	// The default value is the logged-on username of the application's process.
	const wchar_t *userName(void);
	// The default user name to be stored in each TAR header when writing TAR archives.
	// The default value is the logged-on username of the application's process.
	void put_UserName(const wchar_t *newVal);

	// The TAR format to use when writing a TAR archive. Valid values are "gnu", "pax",
	// and "ustar". The default value is "gnu".
	void get_WriteFormat(CkString &str);
	// The TAR format to use when writing a TAR archive. Valid values are "gnu", "pax",
	// and "ustar". The default value is "gnu".
	const wchar_t *writeFormat(void);
	// The TAR format to use when writing a TAR archive. Valid values are "gnu", "pax",
	// and "ustar". The default value is "gnu".
	void put_WriteFormat(const wchar_t *newVal);

	// Contains the XML listing of the contents of the TAR archive for the last untar
	// method call (such as Untar, UntarGz, etc.) where the CaptureXmlListing property
	// was set to true.
	void get_XmlListing(CkString &str);
	// Contains the XML listing of the contents of the TAR archive for the last untar
	// method call (such as Untar, UntarGz, etc.) where the CaptureXmlListing property
	// was set to true.
	const wchar_t *xmlListing(void);
	// Contains the XML listing of the contents of the TAR archive for the last untar
	// method call (such as Untar, UntarGz, etc.) where the CaptureXmlListing property
	// was set to true.
	void put_XmlListing(const wchar_t *newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Adds a directory tree to be included in the next call to one of the WriteTar*
	// methods. To include multiple directory trees in a .tar, call AddDirRoot multiple
	// times followed by a single call to WriteTar.
	bool AddDirRoot(const wchar_t *dirPath);

	// Adds a directory tree to be included in the next call to one of the WriteTar*
	// methods. To include multiple directory trees in a .tar, call AddDirRoot2 (and/or
	// AddDirRoot) multiple times followed by a single call to WriteTar.
	// 
	// The rootPrefix adds a prefix to the path in the TAR for all files added under this
	// root. The rootPrefix should not end with a forward-slash char. For example: This is
	// good: "abc/123", but this is not good: "abc/123/". If the DirPrefix property is
	// also set, its prefix will added first.
	// 
	bool AddDirRoot2(const wchar_t *rootPrefix, const wchar_t *rootPath);

	// Adds a local file to be included in the next call to one of the WriteTar*
	// methods. To include multiple files or directory trees in a .tar, call
	// AddFile/AddDirRoot multiple times followed by a single call to WriteTar (or
	// WriteTarGz, or WriteTarBz2).
	bool AddFile(const wchar_t *path);

	// Adds a local file to be included in the next call to one of the WriteTar*
	// methods. Allows for the path within the TAR to be specified. To include multiple
	// files or directory trees in a .tar, call AddFile/AddFile2/AddDirRoot multiple
	// times followed by a single call to WriteTar (or WriteTarGz, or WriteTarBz2).
	// 
	// Note: The DirPrefix property does not apply to files added via this method
	// because this method explicilty specifies the path-in-tar.
	// 
	bool AddFile2(const wchar_t *filePath, const wchar_t *pathWithinTar);

	// Clears all directory roots and files previously added by calls to AddDirRoot,
	// AddDirRoot2, AddFile, and AddFile2.
	bool ClearDirRootsAndFiles(void);

	// Creates a .deb Debian binary package archive from a control.tar.gz and
	// data.tar.gz. The controlPath is the path to the control.tar.gz file (or equivalent),
	// and the dataPath is the path to the data.tar.gz file. The output file path (.deb) is
	// specified in debPath.
	bool CreateDeb(const wchar_t *controlPath, const wchar_t *dataPath, const wchar_t *debPath);

	// Returns the value of the Nth directory root. For example, if an application
	// calls AddDirRoot twice, then the NumDirRoots property would have a value of 2,
	// and GetDirRoot(0) would return the path passed to AddDirRoot in the 1st call,
	// and GetDirRoot(1) would return the directory path in the 2nd call to AddDirRoot.
	bool GetDirRoot(int index, CkString &outStr);
	// Returns the value of the Nth directory root. For example, if an application
	// calls AddDirRoot twice, then the NumDirRoots property would have a value of 2,
	// and GetDirRoot(0) would return the path passed to AddDirRoot in the 1st call,
	// and GetDirRoot(1) would return the directory path in the 2nd call to AddDirRoot.
	const wchar_t *getDirRoot(int index);
	// Returns the value of the Nth directory root. For example, if an application
	// calls AddDirRoot twice, then the NumDirRoots property would have a value of 2,
	// and GetDirRoot(0) would return the path passed to AddDirRoot in the 1st call,
	// and GetDirRoot(1) would return the directory path in the 2nd call to AddDirRoot.
	const wchar_t *dirRoot(int index);

	// Scans a TAR archive and returns XML detailing the files and directories found
	// within the TAR.
	bool ListXml(const wchar_t *tarPath, CkString &outStr);
	// Scans a TAR archive and returns XML detailing the files and directories found
	// within the TAR.
	const wchar_t *listXml(const wchar_t *tarPath);

	// Creates an asynchronous task to call the ListXml method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ListXmlAsync(const wchar_t *tarPath);

	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTaskW &task);

	// Unlocks the component allowing for the full functionality to be used. If this
	// method unexpectedly returns false, examine the contents of the LastErrorText
	// property to determine the reason for failure.
	bool UnlockComponent(const wchar_t *unlockCode);

	// Extracts the files and directories from a TAR archive, reconstructing the
	// directory tree(s) in the local filesystem. The files are extracted to the
	// directory specified by the UntarFromDir property. Returns the number of files
	// and directories extracted, or -1 for failure.
	int Untar(const wchar_t *tarPath);

	// Creates an asynchronous task to call the Untar method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UntarAsync(const wchar_t *tarPath);

	// Extracts the files and directories from a tar.bz2 (or tar.bzip2) archive,
	// reconstructing the directory tree(s) in the local filesystem. The files are
	// extracted to the directory specified by the UntarFromDir property.
	bool UntarBz2(const wchar_t *tarPath);

	// Creates an asynchronous task to call the UntarBz2 method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UntarBz2Async(const wchar_t *tarPath);

	// Untars the first file matching the matchPattern into bd.
	bool UntarFirstMatchingToBd(const wchar_t *tarPath, const wchar_t *matchPattern, CkBinDataW &bd);

	// Memory-to-memory untar. The first file matching the matchPattern is extracted and
	// returned.
	bool UntarFirstMatchingToMemory(CkByteData &tarFileBytes, const wchar_t *matchPattern, CkByteData &outBytes);

	// Extracts the files and directories from an in-memory TAR archive, reconstructing
	// the directory tree(s) in the local filesystem. The files are extracted to the
	// directory specified by the UntarFromDir property. Returns the number of files
	// and directories extracted, or -1 for failure.
	int UntarFromMemory(CkByteData &tarFileBytes);

	// Creates an asynchronous task to call the UntarFromMemory method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UntarFromMemoryAsync(CkByteData &tarFileBytes);

	// Extracts the files and directories from a tar.gz (or tar.gzip) archive,
	// reconstructing the directory tree(s) in the local filesystem. The files are
	// extracted to the directory specified by the UntarFromDir property.
	bool UntarGz(const wchar_t *tarPath);

	// Creates an asynchronous task to call the UntarGz method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UntarGzAsync(const wchar_t *tarPath);

	// Extracts the files and directories from a tar.Z archive, reconstructing the
	// directory tree(s) in the local filesystem. The files are extracted to the
	// directory specified by the UntarFromDir property.
	bool UntarZ(const wchar_t *tarPath);

	// Creates an asynchronous task to call the UntarZ method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UntarZAsync(const wchar_t *tarPath);

	// Verifies that a TAR archive is valid. This method opens the TAR archive and
	// scans the entire file by walking the TAR headers. Returns true if no errors
	// were found. Otherwise returns false.
	bool VerifyTar(const wchar_t *tarPath);

	// Creates an asynchronous task to call the VerifyTar method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *VerifyTarAsync(const wchar_t *tarPath);

	// Writes a TAR archive. The directory trees previously added by calling AddDirRoot
	// one or more times are included in the output TAR archive.
	bool WriteTar(const wchar_t *tarPath);

	// Creates an asynchronous task to call the WriteTar method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *WriteTarAsync(const wchar_t *tarPath);

	// Writes a .tar.bz2 compressed TAR archive. The directory trees previously added
	// by calling AddDirRoot one or more times are included in the output file.
	bool WriteTarBz2(const wchar_t *bz2Path);

	// Creates an asynchronous task to call the WriteTarBz2 method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *WriteTarBz2Async(const wchar_t *bz2Path);

	// Writes a .tar.gz (also known as .tgz) compressed TAR archive. The directory
	// trees previously added by calling AddDirRoot one or more times are included in
	// the output file.
	bool WriteTarGz(const wchar_t *gzPath);

	// Creates an asynchronous task to call the WriteTarGz method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *WriteTarGzAsync(const wchar_t *gzPath);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
