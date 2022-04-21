// CkZipCrcW.h: interface for the CkZipCrcW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkZipCrcW_H
#define _CkZipCrcW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacksW.h"

class CkByteData;
class CkBinDataW;
class CkStringBuilderW;
class CkTaskW;
class CkBaseProgressW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkZipCrcW
class CK_VISIBLE_PUBLIC CkZipCrcW  : public CkClassWithCallbacksW
{
	private:
	bool m_cbOwned;

	private:
	
	// Don't allow assignment or copying these objects.
	CkZipCrcW(const CkZipCrcW &);
	CkZipCrcW &operator=(const CkZipCrcW &);

    public:
	CkZipCrcW(void);
	virtual ~CkZipCrcW(void);

	

	static CkZipCrcW *createNew(void);
	

	CkZipCrcW(bool bCallbackOwned);
	static CkZipCrcW *createNew(bool bCallbackOwned);

	
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


	// ----------------------
	// Methods
	// ----------------------
	// Provides a way to calculate a CRC by streaming the data a chunk at a time. An
	// application would start by calling BeginStream. Then it would add data by
	// calling MoreData for each additional chunk. After the last chunk has been
	// processed, the EndStream method is called to return the CRC.
	void BeginStream(void);

	// Calculates a 32-bit CRC for in-memory byte data. This is the 32-bit CRC that
	// would be found in a Zip file header if a file containing the data was added to a
	// zip archive. Returns the CRC32 of the data.
	unsigned long CalculateCrc(CkByteData &data);

	// Calculates a CRC32 for the bytes contained in bd.
	unsigned long CrcBd(CkBinDataW &bd);

	// Calculates a CRC32 for the string contained in sb. The charset is the byte
	// representation to be used for the sb when calculating the CRC32. It can be
	// utf-8, utf-16, windows-1252, iso-8859-1, or any of the character encodings
	// (charsets) listed at the link below.
	unsigned long CrcSb(CkStringBuilderW &sb, const wchar_t *charset);

	// Calculates a CRC32 for a string. The charset is the byte representation to be used
	// for the str when calculating the CRC32. It can be utf-8, utf-16, windows-1252,
	// iso-8859-1, or any of the character encodings (charsets) listed at the link
	// below.
	unsigned long CrcString(const wchar_t *str, const wchar_t *charset);

	// Finalizes and returns the Zip CRC value calculated by calling BeginStream
	// followed by multiple calls to MoreData.
	unsigned long EndStream(void);

	// Calculates the CRC32 of a file. The data contained in the file is streamed for
	// the calculation to keep the memory footprint small and constant. Returns the
	// CRC32 of the file.
	unsigned long FileCrc(const wchar_t *path);

	// Creates an asynchronous task to call the FileCrc method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *FileCrcAsync(const wchar_t *path);

	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTaskW &task);

	// Adds additional data to the CRC currently being calculated. (See BeginStream for
	// more information.)
	void MoreData(CkByteData &data);

	// Converts a 32-bit integer to a hex string.
	bool ToHex(unsigned long crc, CkString &outStr);
	// Converts a 32-bit integer to a hex string.
	const wchar_t *toHex(unsigned long crc);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
