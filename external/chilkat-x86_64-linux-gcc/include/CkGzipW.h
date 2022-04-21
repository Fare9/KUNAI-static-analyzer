// CkGzipW.h: interface for the CkGzipW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkGzipW_H
#define _CkGzipW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacksW.h"

class CkByteData;
class CkBinDataW;
class CkTaskW;
class CkDateTimeW;
class CkBaseProgressW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkGzipW
class CK_VISIBLE_PUBLIC CkGzipW  : public CkClassWithCallbacksW
{
	private:
	bool m_cbOwned;

	private:
	
	// Don't allow assignment or copying these objects.
	CkGzipW(const CkGzipW &);
	CkGzipW &operator=(const CkGzipW &);

    public:
	CkGzipW(void);
	virtual ~CkGzipW(void);

	

	static CkGzipW *createNew(void);
	

	CkGzipW(bool bCallbackOwned);
	static CkGzipW *createNew(bool bCallbackOwned);

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	CkBaseProgressW *get_EventCallbackObject(void) const;
	void put_EventCallbackObject(CkBaseProgressW *progress);


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

	// Specifies an optional comment string that can be embedded within the .gz when
	// any Compress* method is called.
	void get_Comment(CkString &str);
	// Specifies an optional comment string that can be embedded within the .gz when
	// any Compress* method is called.
	const wchar_t *comment(void);
	// Specifies an optional comment string that can be embedded within the .gz when
	// any Compress* method is called.
	void put_Comment(const wchar_t *newVal);

	// The compression level to be used when compressing. The default is 6, which is
	// the typical value used for zip utility programs when compressing data. The
	// compression level may range from 0 (no compression) to 9 (the most compression).
	// The benefits of trying to increase compression may not be worth the added
	// expense in compression time. In many cases (which is data dependent), the
	// improvement in compression is minimal while the increase in computation time is
	// significant.
	int get_CompressionLevel(void);
	// The compression level to be used when compressing. The default is 6, which is
	// the typical value used for zip utility programs when compressing data. The
	// compression level may range from 0 (no compression) to 9 (the most compression).
	// The benefits of trying to increase compression may not be worth the added
	// expense in compression time. In many cases (which is data dependent), the
	// improvement in compression is minimal while the increase in computation time is
	// significant.
	void put_CompressionLevel(int newVal);

	// Optional extra-data that can be included within a .gz when a Compress* method is
	// called.
	void get_ExtraData(CkByteData &outBytes);
	// Optional extra-data that can be included within a .gz when a Compress* method is
	// called.
	void put_ExtraData(const CkByteData &inBytes);

	// The filename that is embedded within the .gz during any Compress* method call.
	// When extracting from a .gz using applications such as WinZip, this will be the
	// filename that is created.
	void get_Filename(CkString &str);
	// The filename that is embedded within the .gz during any Compress* method call.
	// When extracting from a .gz using applications such as WinZip, this will be the
	// filename that is created.
	const wchar_t *filename(void);
	// The filename that is embedded within the .gz during any Compress* method call.
	// When extracting from a .gz using applications such as WinZip, this will be the
	// filename that is created.
	void put_Filename(const wchar_t *newVal);

	// The number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any method call prior to
	// completion. If HeartbeatMs is 0 (the default), no AbortCheck event callbacks
	// will fire.
	int get_HeartbeatMs(void);
	// The number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any method call prior to
	// completion. If HeartbeatMs is 0 (the default), no AbortCheck event callbacks
	// will fire.
	void put_HeartbeatMs(int newVal);

	// The last-modification date/time to be embedded within the .gz when a Compress*
	// method is called. By default, the current system date/time is used.
	void get_LastMod(SYSTEMTIME &outSysTime);
	// The last-modification date/time to be embedded within the .gz when a Compress*
	// method is called. By default, the current system date/time is used.
	void put_LastMod(const SYSTEMTIME &sysTime);

	// The same as the LastMod property, but allows the date/time to be get/set in
	// RFC822 string format.
	void get_LastModStr(CkString &str);
	// The same as the LastMod property, but allows the date/time to be get/set in
	// RFC822 string format.
	const wchar_t *lastModStr(void);
	// The same as the LastMod property, but allows the date/time to be get/set in
	// RFC822 string format.
	void put_LastModStr(const wchar_t *newVal);

	// If set to true, the file produced by an Uncompress* method will use the
	// current date/time for the last-modification date instead of the date/time found
	// within the Gzip format.
	bool get_UseCurrentDate(void);
	// If set to true, the file produced by an Uncompress* method will use the
	// current date/time for the last-modification date instead of the date/time found
	// within the Gzip format.
	void put_UseCurrentDate(bool newVal);



	// ----------------------
	// Methods
	// ----------------------
	// In-place gzip the contents of binDat.
	bool CompressBd(CkBinDataW &binDat);

	// Creates an asynchronous task to call the CompressBd method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CompressBdAsync(CkBinDataW &binDat);

	// Compresses a file to create a GZip compressed file (.gz).
	bool CompressFile(const wchar_t *inFilename, const wchar_t *destPath);

	// Creates an asynchronous task to call the CompressFile method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CompressFileAsync(const wchar_t *inFilename, const wchar_t *destPath);

	// Compresses a file to create a GZip compressed file (.gz). inFilename is the actual
	// filename on disk. embeddedFilename is the filename to be embedded in the .gz such that when
	// it is un-gzipped, this is the name of the file that will be created.
	bool CompressFile2(const wchar_t *inFilename, const wchar_t *embeddedFilename, const wchar_t *destPath);

	// Creates an asynchronous task to call the CompressFile2 method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CompressFile2Async(const wchar_t *inFilename, const wchar_t *embeddedFilename, const wchar_t *destPath);

	// Gzip compresses a file to an in-memory image of a .gz file.
	// 
	// Note: There is a 4GB size limitation. The compressed size of the file cannot be
	// more than 4GB. Chilkat will be working to alleviate this limitation in the
	// future.
	// 
	bool CompressFileToMem(const wchar_t *inFilename, CkByteData &outData);

	// Creates an asynchronous task to call the CompressFileToMem method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CompressFileToMemAsync(const wchar_t *inFilename);

	// Compresses in-memory data to an in-memory image of a .gz file.
	// 
	// Note: There is a 4GB uncompressed size limitation. The uncompressed size of the
	// file cannot be more than 4GB. Chilkat will be working to alleviate this
	// limitation in the future.
	// 
	bool CompressMemory(CkByteData &inData, CkByteData &outData);

	// Creates an asynchronous task to call the CompressMemory method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CompressMemoryAsync(CkByteData &inData);

	// Gzip compresses and creates a .gz file from in-memory data.
	bool CompressMemToFile(CkByteData &inData, const wchar_t *destPath);

	// Creates an asynchronous task to call the CompressMemToFile method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CompressMemToFileAsync(CkByteData &inData, const wchar_t *destPath);

	// Gzip compresses a string and writes the output to a byte array. The string is
	// first converted to the charset specified by destCharset. Typical charsets are "utf-8",
	// "iso-8859-1", "shift_JIS", etc.
	bool CompressString(const wchar_t *inStr, const wchar_t *destCharset, CkByteData &outBytes);

	// Creates an asynchronous task to call the CompressString method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CompressStringAsync(const wchar_t *inStr, const wchar_t *destCharset);

	// The same as CompressString, except the binary output is returned in encoded
	// string form according to the encoding. The encoding can be any of the following:
	// "Base64", "modBase64", "Base32", "UU", "QP" (for quoted-printable), "URL" (for
	// url-encoding), "Hex", "Q", "B", "url_oath", "url_rfc1738", "url_rfc2396", and
	// "url_rfc3986".
	bool CompressStringENC(const wchar_t *inStr, const wchar_t *charset, const wchar_t *encoding, CkString &outStr);
	// The same as CompressString, except the binary output is returned in encoded
	// string form according to the encoding. The encoding can be any of the following:
	// "Base64", "modBase64", "Base32", "UU", "QP" (for quoted-printable), "URL" (for
	// url-encoding), "Hex", "Q", "B", "url_oath", "url_rfc1738", "url_rfc2396", and
	// "url_rfc3986".
	const wchar_t *compressStringENC(const wchar_t *inStr, const wchar_t *charset, const wchar_t *encoding);

	// Gzip compresses a string and writes the output to a .gz compressed file. The
	// string is first converted to the charset specified by destCharset. Typical charsets are
	// "utf-8", "iso-8859-1", "shift_JIS", etc.
	bool CompressStringToFile(const wchar_t *inStr, const wchar_t *destCharset, const wchar_t *destPath);

	// Creates an asynchronous task to call the CompressStringToFile method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CompressStringToFileAsync(const wchar_t *inStr, const wchar_t *destCharset, const wchar_t *destPath);

	// Decodes an encoded string and returns the original data. The encoding mode is
	// determined by encoding. It may be "base64", "hex", "quoted-printable", or "url".
	bool Decode(const wchar_t *encodedStr, const wchar_t *encoding, CkByteData &outBytes);

	// Provides the ability to use the zip/gzip's deflate algorithm directly to
	// compress a string. Internal to this method, inString is first converted to the
	// charset specified by charsetName. The data is then compressed using the deflate
	// compression algorithm. The binary output is then encoded according to outputEncoding.
	// Possible values for outputEncoding are "hex", "base64", "url", and "quoted-printable".
	// 
	// Note: The output of this method is compressed data with no Gzip file format. Use
	// the Compress* methods to produce Gzip file format output.
	// 
	bool DeflateStringENC(const wchar_t *inString, const wchar_t *charsetName, const wchar_t *outputEncoding, CkString &outStr);
	// Provides the ability to use the zip/gzip's deflate algorithm directly to
	// compress a string. Internal to this method, inString is first converted to the
	// charset specified by charsetName. The data is then compressed using the deflate
	// compression algorithm. The binary output is then encoded according to outputEncoding.
	// Possible values for outputEncoding are "hex", "base64", "url", and "quoted-printable".
	// 
	// Note: The output of this method is compressed data with no Gzip file format. Use
	// the Compress* methods to produce Gzip file format output.
	// 
	const wchar_t *deflateStringENC(const wchar_t *inString, const wchar_t *charsetName, const wchar_t *outputEncoding);

	// Encodes binary data to a printable string. The encoding mode is determined by
	// encoding. It may be "base64", "hex", "quoted-printable", or "url".
	bool Encode(CkByteData &byteData, const wchar_t *encoding, CkString &outStr);
	// Encodes binary data to a printable string. The encoding mode is determined by
	// encoding. It may be "base64", "hex", "quoted-printable", or "url".
	const wchar_t *encode(CkByteData &byteData, const wchar_t *encoding);

	// Determines if the inGzFilename is a Gzip formatted file. Returns true if it is a Gzip
	// formatted file, otherwise returns false.
	bool ExamineFile(const wchar_t *inGzFilename);

	// Determines if the in-memory bytes (inGzData) contain a Gzip formatted file. Returns
	// true if it is Gzip format, otherwise returns false.
	bool ExamineMemory(CkByteData &inGzData);

	// Gets the last-modification date/time to be embedded within the .gz.
	// The caller is responsible for deleting the object returned by this method.
	CkDateTimeW *GetDt(void);

	// This the reverse of DeflateStringENC.
	// 
	// The input string is first decoded according to inputEncoding. (Possible values for inputEncoding
	// are "hex", "base64", "url", and "quoted-printable".) The compressed data is then
	// inflated, and the result is then converted from convertFromCharset (if necessary) to return a
	// string.
	// 
	bool InflateStringENC(const wchar_t *inString, const wchar_t *convertFromCharset, const wchar_t *inputEncoding, CkString &outStr);
	// This the reverse of DeflateStringENC.
	// 
	// The input string is first decoded according to inputEncoding. (Possible values for inputEncoding
	// are "hex", "base64", "url", and "quoted-printable".) The compressed data is then
	// inflated, and the result is then converted from convertFromCharset (if necessary) to return a
	// string.
	// 
	const wchar_t *inflateStringENC(const wchar_t *inString, const wchar_t *convertFromCharset, const wchar_t *inputEncoding);

	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTaskW &task);

	// Reads a binary file into memory and returns the byte array. Note: This method
	// does not parse a Gzip, it is only a convenience method for reading a binary file
	// into memory.
	bool ReadFile(const wchar_t *path, CkByteData &outBytes);

	// Sets the last-modification date/time to be embedded within the .gz when a
	// Compress* method is called. If not explicitly set, the current system date/time
	// is used.
	bool SetDt(CkDateTimeW &dt);

	// In-place ungzip the contents of binDat.
	bool UncompressBd(CkBinDataW &binDat);

	// Creates an asynchronous task to call the UncompressBd method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UncompressBdAsync(CkBinDataW &binDat);

	// Un-Gzips a .gz file. The output filename is specified by the 2nd argument and
	// not by the filename embedded within the .gz.
	bool UncompressFile(const wchar_t *srcPath, const wchar_t *destPath);

	// Creates an asynchronous task to call the UncompressFile method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UncompressFileAsync(const wchar_t *srcPath, const wchar_t *destPath);

	// Un-Gzips a .gz file directly to memory.
	// 
	// Note: There is a 4GB uncompressed size limitation. The uncompressed size of the
	// file cannot be more than 4GB. Chilkat will be working to alleviate this
	// limitation in the future.
	// 
	bool UncompressFileToMem(const wchar_t *inFilename, CkByteData &outData);

	// Creates an asynchronous task to call the UncompressFileToMem method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UncompressFileToMemAsync(const wchar_t *inFilename);

	// Uncompresses a .gz file that contains a text file. The contents of the text file
	// are returned as a string. The character encoding of the text file is specified
	// by charset. Typical charsets are "iso-8859-1", "utf-8", "windows-1252",
	// "shift_JIS", "big5", etc.
	bool UncompressFileToString(const wchar_t *gzFilename, const wchar_t *charset, CkString &outStr);
	// Uncompresses a .gz file that contains a text file. The contents of the text file
	// are returned as a string. The character encoding of the text file is specified
	// by charset. Typical charsets are "iso-8859-1", "utf-8", "windows-1252",
	// "shift_JIS", "big5", etc.
	const wchar_t *uncompressFileToString(const wchar_t *gzFilename, const wchar_t *charset);

	// Creates an asynchronous task to call the UncompressFileToString method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UncompressFileToStringAsync(const wchar_t *gzFilename, const wchar_t *charset);

	// Un-Gzips from an in-memory image of a .gz file directly into memory.
	// 
	// Note: There is a 4GB uncompressed size limitation. The uncompressed size of the
	// file cannot be more than 4GB. Chilkat will be working to alleviate this
	// limitation in the future.
	// 
	bool UncompressMemory(CkByteData &inData, CkByteData &outData);

	// Creates an asynchronous task to call the UncompressMemory method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UncompressMemoryAsync(CkByteData &inData);

	// Un-Gzips from an in-memory image of a .gz file to a file.
	bool UncompressMemToFile(CkByteData &inData, const wchar_t *destPath);

	// Creates an asynchronous task to call the UncompressMemToFile method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UncompressMemToFileAsync(CkByteData &inData, const wchar_t *destPath);

	// The reverse of CompressString.
	// 
	// The bytes in inData are uncompressed, then converted from inCharset (if necessary) to
	// return a string.
	// 
	bool UncompressString(CkByteData &inData, const wchar_t *inCharset, CkString &outStr);
	// The reverse of CompressString.
	// 
	// The bytes in inData are uncompressed, then converted from inCharset (if necessary) to
	// return a string.
	// 
	const wchar_t *uncompressString(CkByteData &inData, const wchar_t *inCharset);

	// Creates an asynchronous task to call the UncompressString method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UncompressStringAsync(CkByteData &inData, const wchar_t *inCharset);

	// The same as UncompressString, except the compressed data is provided in encoded
	// string form based on the value of encoding. The encoding can be "Base64", "modBase64",
	// "Base32", "UU", "QP" (for quoted-printable), "URL" (for url-encoding), "Hex",
	// "Q", "B", "url_oath", "url_rfc1738", "url_rfc2396", and "url_rfc3986".
	bool UncompressStringENC(const wchar_t *inStr, const wchar_t *charset, const wchar_t *encoding, CkString &outStr);
	// The same as UncompressString, except the compressed data is provided in encoded
	// string form based on the value of encoding. The encoding can be "Base64", "modBase64",
	// "Base32", "UU", "QP" (for quoted-printable), "URL" (for url-encoding), "Hex",
	// "Q", "B", "url_oath", "url_rfc1738", "url_rfc2396", and "url_rfc3986".
	const wchar_t *uncompressStringENC(const wchar_t *inStr, const wchar_t *charset, const wchar_t *encoding);

	// Unlocks the component allowing for the full functionality to be used.
	bool UnlockComponent(const wchar_t *unlockCode);

	// Unpacks a .tar.gz file. The ungzip and untar occur in streaming mode. There are
	// no temporary files and the memory footprint is constant (and small) while
	// untarring. bNoAbsolute may be true or false. A value of true protects from
	// untarring to absolute paths. (For example, imagine the trouble if the tar
	// contains files with absolute paths beginning with /Windows/system32)
	bool UnTarGz(const wchar_t *tgzFilename, const wchar_t *destDir, bool bNoAbsolute);

	// Creates an asynchronous task to call the UnTarGz method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UnTarGzAsync(const wchar_t *tgzFilename, const wchar_t *destDir, bool bNoAbsolute);

	// A convenience method for writing a binary byte array to a file.
	bool WriteFile(const wchar_t *path, CkByteData &binaryData);

	// Converts base64-gzip .xfdl data to a decompressed XML string. The xfldData contains
	// the base64 data. This method returns the decoded/decompressed XML string.
	bool XfdlToXml(const wchar_t *xfldData, CkString &outStr);
	// Converts base64-gzip .xfdl data to a decompressed XML string. The xfldData contains
	// the base64 data. This method returns the decoded/decompressed XML string.
	const wchar_t *xfdlToXml(const wchar_t *xfldData);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
