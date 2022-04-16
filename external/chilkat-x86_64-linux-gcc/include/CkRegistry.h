
//
// This is NOT a generated or documented Chilkat class.
// This is NOT a generated or documented Chilkat class.
// This is NOT a generated or documented Chilkat class.
// This is NOT a generated or documented Chilkat class.
// This is NOT a generated or documented Chilkat class.
// This is NOT a generated or documented Chilkat class.
//

#ifndef _CkRegistry_H
#define _CkRegistry_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

// CLASS: CkRegistry
class CK_VISIBLE_PUBLIC CkRegistry  : public CkMultiByteBase
{
    private:
	
	// Don't allow assignment or copying these objects.
	CkRegistry(const CkRegistry &);
	CkRegistry &operator=(const CkRegistry &);

    public:
	CkRegistry(void);
	virtual ~CkRegistry(void);

	static CkRegistry *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
	// BEGIN PUBLIC INTERFACE
        bool SetProductInfo(const char *company,
            const char *productName,
            const char *keyName,
            const char *value);

        bool GetProductInfo(const char *company,
            const char *productName,
            const char *keyName,
            CkString &keyValue);

	const char *getProductInfo(const char *company,
            const char *productName,
            const char *keyName);


	// END PUBLIC INTERFACE


};

	
#endif
