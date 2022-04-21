// CkZipEntry.h: interface for the CkZipEntry class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkZipEntry_H
#define _CkZipEntry_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacks.h"

class CkTask;
class CkByteData;
class CkDateTime;
class CkBinData;
class CkStringBuilder;
class CkStream;
class CkBaseProgress;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkZipEntry
class CK_VISIBLE_PUBLIC CkZipEntry  : public CkClassWithCallbacks
{
    private:

	// Don't allow assignment or copying these objects.
	CkZipEntry(const CkZipEntry &);
	CkZipEntry &operator=(const CkZipEntry &);

    public:
	CkZipEntry(void);
	virtual ~CkZipEntry(void);

	static CkZipEntry *createNew(void);
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
	// The comment stored within the Zip for this entry.
	void get_Comment(CkString &str);
	// The comment stored within the Zip for this entry.
	const char *comment(void);
	// The comment stored within the Zip for this entry.
	void put_Comment(const char *newVal);

	// The size in bytes of this entry's file data when compression is applied
	unsigned long get_CompressedLength(void);

	// The size in bytes of this entry's file data when compression is applied
	__int64 get_CompressedLength64(void);

	// The size in decimal string format of this file when Zip compression is applied.
	void get_CompressedLengthStr(CkString &str);
	// The size in decimal string format of this file when Zip compression is applied.
	const char *compressedLengthStr(void);

	// The compression level. 0 = no compression, while 9 = maximum compression. The
	// default is 6.
	int get_CompressionLevel(void);
	// The compression level. 0 = no compression, while 9 = maximum compression. The
	// default is 6.
	void put_CompressionLevel(int newVal);

	// Set to 0 for no compression, or 8 for the Deflate algorithm. The Deflate
	// algorithm is the default algorithm of the most popular Zip utility programs,
	// such as WinZip
	int get_CompressionMethod(void);
	// Set to 0 for no compression, or 8 for the Deflate algorithm. The Deflate
	// algorithm is the default algorithm of the most popular Zip utility programs,
	// such as WinZip
	void put_CompressionMethod(int newVal);

	// The CRC for the zip entry. For AES encrypted entries, the CRC value will be 0.
	// (See http://www.winzip.com/aes_info.htm#CRC )
	int get_Crc(void);

	// If this entry is AES encrypted, then this property contains the AES key length
	// (128, 192, or 256). If the entry is not AES encrypted, then the value is 0.
	int get_EncryptionKeyLen(void);

	// A unique ID assigned to the entry while the object is instantiated in memory.
	int get_EntryID(void);

	// Indicates the origin of the entry. There are three possible values:
	//     Mapped Entry: An entry in an existing Zip file.
	//     File Entry: A file not yet in memory, but referenced. These entries are
	//     added by calling AppendFiles, AppendFilesEx, AppendOneFileOrDir, etc.
	//     Data Entry: An entry containing uncompressed data from memory. These entries
	//     are added by calling AppendData, AppendString, etc.
	//     Null Entry: An entry that no longer exists in the .zip.
	//     New Directory Entry: A directory entry added by calling AppendNewDir.
	// When the zip is written by calling WriteZip or WriteToMemory, all of the zip
	// entries are transformed into mapped entries. They become entries that contain
	// the compressed data within the .zip that was just created. (The WriteZip method
	// call effectively writes the zip and then opens it, replacing all of the existing
	// entries with mapped entries.)
	int get_EntryType(void);

	// The local last-modified date/time.
	void get_FileDateTime(SYSTEMTIME &outSysTime);
	// The local last-modified date/time.
	void put_FileDateTime(const SYSTEMTIME &sysTime);

	// The local last-modified date/time in RFC822 string format.
	void get_FileDateTimeStr(CkString &str);
	// The local last-modified date/time in RFC822 string format.
	const char *fileDateTimeStr(void);
	// The local last-modified date/time in RFC822 string format.
	void put_FileDateTimeStr(const char *newVal);

	// The file name of the Zip entry.
	void get_FileName(CkString &str);
	// The file name of the Zip entry.
	const char *fileName(void);
	// The file name of the Zip entry.
	void put_FileName(const char *newVal);

	// A string containing the hex encoded raw filename bytes found in the Zip entry.
	void get_FileNameHex(CkString &str);
	// A string containing the hex encoded raw filename bytes found in the Zip entry.
	const char *fileNameHex(void);

	// The number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort inflate/extract/unzip calls
	// prior to completion. If HeartbeatMs is 0 (the default), no AbortCheck event
	// callbacks will fire.
	int get_HeartbeatMs(void);
	// The number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort inflate/extract/unzip calls
	// prior to completion. If HeartbeatMs is 0 (the default), no AbortCheck event
	// callbacks will fire.
	void put_HeartbeatMs(int newVal);

	// true if the entry is AES encrypted. This property can only be true for
	// entries already contained in a .zip (i.e. entries obtained from a zip archive
	// that was opened via OpenZip, OpenBd, OpenFromMemory, etc.) The property is
	// false if the entry contained in the zip is not AES encrypted.
	bool get_IsAesEncrypted(void);

	// True if the Zip entry is a directory, false if it is a file.
	bool get_IsDirectory(void);

	// Controls whether the "text flag" of the internal file attributes for this entry
	// within the zip is set. This is a bit flag that indicates whether the file
	// contents are text or binary. It is for informational use and it is not
	// imperative that this bit flag is accurately set. The ability to set this bit
	// flag is only provided to satisfy certain cases where another application might
	// be sensitive to the flag.
	bool get_TextFlag(void);
	// Controls whether the "text flag" of the internal file attributes for this entry
	// within the zip is set. This is a bit flag that indicates whether the file
	// contents are text or binary. It is for informational use and it is not
	// imperative that this bit flag is accurately set. The ability to set this bit
	// flag is only provided to satisfy certain cases where another application might
	// be sensitive to the flag.
	void put_TextFlag(bool newVal);

	// The size in bytes of this entry's file data when uncompressed.
	unsigned long get_UncompressedLength(void);

	// The size in bytes of this entry's file data when uncompressed.
	__int64 get_UncompressedLength64(void);

	// The size in bytes (in decimal string format) of this zip entry's data when
	// uncompressed.
	void get_UncompressedLengthStr(CkString &str);
	// The size in bytes (in decimal string format) of this zip entry's data when
	// uncompressed.
	const char *uncompressedLengthStr(void);



	// ----------------------
	// Methods
	// ----------------------
	// Appends binary data to the zip entry's file contents.
	bool AppendData(CkByteData &inData);

	// Appends binary data to the zip entry's file contents.
	CkTask *AppendDataAsync(CkByteData &inData);


	// Appends text data to the zip entry's file contents. The text is appended using
	// the character encoding specified by the charset, which can be "utf-8", "ansi", etc.
	bool AppendString(const char *strContent, const char *charset);

	// Appends text data to the zip entry's file contents. The text is appended using
	// the character encoding specified by the charset, which can be "utf-8", "ansi", etc.
	CkTask *AppendStringAsync(const char *strContent, const char *charset);


	// Returns the compressed data as a byte array.
	// 
	// Note: The Copy method can only be called if the zip entry already contains
	// compressed data (i.e. it is a "mapped entry"). This is the case when an existing
	// .zip is opened (from memory or from a file), or after the .zip has been written
	// (by calling WriteZip or WriteToMemory). If a zip entry is created via
	// AppendData, AppendFiles, etc., then it does not yet contain compressed data.
	// When the zip is written, each entry becomes a "mapped entry" (The value of the
	// EntryType property becomes 0.)
	// 
	bool Copy(CkByteData &outData);


	// Returns the compressed data as a Base64-encoded string. It is only possible to
	// retrieve the compressed data from a pre-existing .zip that has been opened or
	// after writing the .zip but not closing it.
	// 
	// Note: The CopyToBase64 method can only be called if the zip entry already
	// contains compressed data (i.e. it is a "mapped entry").
	// 
	bool CopyToBase64(CkString &outStr);

	// Returns the compressed data as a Base64-encoded string. It is only possible to
	// retrieve the compressed data from a pre-existing .zip that has been opened or
	// after writing the .zip but not closing it.
	// 
	// Note: The CopyToBase64 method can only be called if the zip entry already
	// contains compressed data (i.e. it is a "mapped entry").
	// 
	const char *copyToBase64(void);

	// Returns the compressed data as a hexidecimal encoded string. It is only possible
	// to retrieve the compressed data from a pre-existing .zip that has been opened or
	// after writing the .zip but not closing it.
	// 
	// Note: The CopyToBase64 method can only be called if the zip entry already
	// contains compressed data (i.e. it is a "mapped entry").
	// 
	bool CopyToHex(CkString &outStr);

	// Returns the compressed data as a hexidecimal encoded string. It is only possible
	// to retrieve the compressed data from a pre-existing .zip that has been opened or
	// after writing the .zip but not closing it.
	// 
	// Note: The CopyToBase64 method can only be called if the zip entry already
	// contains compressed data (i.e. it is a "mapped entry").
	// 
	const char *copyToHex(void);

	// Unzips this entry into the specified base directory. The file is unzipped to the
	// subdirectory according to the relative path stored with the entry in the Zip.
	// Use ExtractInto to unzip into a specific directory regardless of the path
	// information stored in the Zip.
	bool Extract(const char *dirPath);

	// Unzips this entry into the specified base directory. The file is unzipped to the
	// subdirectory according to the relative path stored with the entry in the Zip.
	// Use ExtractInto to unzip into a specific directory regardless of the path
	// information stored in the Zip.
	CkTask *ExtractAsync(const char *dirPath);


	// Unzip a file into a specific directory. If this entry is a directory, then
	// nothing occurs. (An application can check the IsDirectory property and instead
	// call Extract if it is desired to create the directory. )
	bool ExtractInto(const char *dirPath);

	// Unzip a file into a specific directory. If this entry is a directory, then
	// nothing occurs. (An application can check the IsDirectory property and instead
	// call Extract if it is desired to create the directory. )
	CkTask *ExtractIntoAsync(const char *dirPath);


	// Returns the last-modified date/time of this zip entry.
	// The caller is responsible for deleting the object returned by this method.
	CkDateTime *GetDt(void);


	// Inflate a file within a Zip directly into memory.
	bool Inflate(CkByteData &outData);

	// Inflate a file within a Zip directly into memory.
	CkTask *InflateAsync(void);


	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTask &task);


	// Return the next entry (file or directory) within the Zip
	// The caller is responsible for deleting the object returned by this method.
	CkZipEntry *NextEntry(void);


	// Returns the next entry having a filename matching the matchStr. The "*" characters
	// matches 0 or more of any character. The full filename, including path, is used
	// when matching against the pattern.
	// The caller is responsible for deleting the object returned by this method.
	CkZipEntry *NextMatchingEntry(const char *matchStr);


	// Replaces the zip entry's existing contents with new data.
	bool ReplaceData(CkByteData &inData);


	// Replaces the zip entry's existing contents with new text data. The text will be
	// stored using the character encoding as specified by charset, which can be "utf-8",
	// "ansi", etc.
	bool ReplaceString(const char *strContent, const char *charset);


	// Sets the last-modified date/time for this zip entry.
	void SetDt(CkDateTime &dt);


	// Unzips the entry contents into the binData.
	bool UnzipToBd(CkBinData &binData);

	// Unzips the entry contents into the binData.
	CkTask *UnzipToBdAsync(CkBinData &binData);


	// Unzips a text file to the sb. The contents of sb are appended with the
	// unzipped file. The lineEndingBehavior is as follows:
	// 
	// 0 = leave unchanged.
	// 1 = convert all to bare LF's
	// 2 = convert all to CRLF's
	// 
	// The srcCharset tells the component how to interpret the bytes of the uncompressed file
	// -- i.e. as utf-8, utf-16, windows-1252, etc.
	bool UnzipToSb(int lineEndingBehavior, const char *srcCharset, CkStringBuilder &sb);

	// Unzips a text file to the sb. The contents of sb are appended with the
	// unzipped file. The lineEndingBehavior is as follows:
	// 
	// 0 = leave unchanged.
	// 1 = convert all to bare LF's
	// 2 = convert all to CRLF's
	// 
	// The srcCharset tells the component how to interpret the bytes of the uncompressed file
	// -- i.e. as utf-8, utf-16, windows-1252, etc.
	CkTask *UnzipToSbAsync(int lineEndingBehavior, const char *srcCharset, CkStringBuilder &sb);


	// Unzips a file within a Zip to a stream. If called synchronously, the toStream must
	// have a sink, such as a file or another stream object. If called asynchronously,
	// then the foreground thread can read the stream.
	bool UnzipToStream(CkStream &toStream);

	// Unzips a file within a Zip to a stream. If called synchronously, the toStream must
	// have a sink, such as a file or another stream object. If called asynchronously,
	// then the foreground thread can read the stream.
	CkTask *UnzipToStreamAsync(CkStream &toStream);


	// Inflate and return the uncompressed data as a string The lineEndingBehavior is as follows:
	// 
	// 0 = leave unchanged.
	// 1 = convert all to bare LF's
	// 2 = convert all to CRLF's
	// 
	// The srcCharset tells the component how to interpret the bytes of the uncompressed file
	// -- i.e. as utf-8, utf-16, windows-1252, etc.
	bool UnzipToString(int lineEndingBehavior, const char *srcCharset, CkString &outStr);

	// Inflate and return the uncompressed data as a string The lineEndingBehavior is as follows:
	// 
	// 0 = leave unchanged.
	// 1 = convert all to bare LF's
	// 2 = convert all to CRLF's
	// 
	// The srcCharset tells the component how to interpret the bytes of the uncompressed file
	// -- i.e. as utf-8, utf-16, windows-1252, etc.
	const char *unzipToString(int lineEndingBehavior, const char *srcCharset);
	// Inflate and return the uncompressed data as a string The lineEndingBehavior is as follows:
	// 
	// 0 = leave unchanged.
	// 1 = convert all to bare LF's
	// 2 = convert all to CRLF's
	// 
	// The srcCharset tells the component how to interpret the bytes of the uncompressed file
	// -- i.e. as utf-8, utf-16, windows-1252, etc.
	CkTask *UnzipToStringAsync(int lineEndingBehavior, const char *srcCharset);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
