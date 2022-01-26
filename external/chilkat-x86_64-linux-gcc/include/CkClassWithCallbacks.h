#ifndef _CkClassWithCallbacks_H
#define _CkClassWithCallbacks_H
#pragma once

#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif

#include "CkMultiByteBase.h"

class CkString;

class CK_VISIBLE_PUBLIC CkClassWithCallbacks : public CkMultiByteBase
{
    protected:
	void *m_callback;
	int m_callbackObjType;

    public:
	    
	CkClassWithCallbacks(void);
	virtual ~CkClassWithCallbacks(void);
    
	// These methods are for internal use only.
	void _setEventCallbackObj(void *p, int objType);
	void *_getEventCallbackObj(void);


};

#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


#endif
	
