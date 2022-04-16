// CkMailManProgress.h: interface for the CkMailManProgress class.
//
//////////////////////////////////////////////////////////////////////

#ifndef _CkMailManProgress_H
#define _CkMailManProgress_H


#include "CkBaseProgress.h"

/*

  To receive progress events (callbacks), create a C++ class that 
  inherits this one and provides one or more overriding implementations 
  for the events you wish to receive.  

  */

// When creating an application class that inherits the CkMailManProgress base class, use the CK_MAILPROGRESS_API 
// definition to declare the overrides in the class header.  This has the effect that if for
// some unforeseen and unlikely reason the Chilkat event callback API changes, or if new
// callback methods are added in a future version, then you'll discover them at compile time
// after updating to the new Chilkat version.  
// For example:
/*
    class MyProgress : public CkMailManProgress
    {
	public:
	    CK_MAILPROGRESS_API

	...
    };
*/
#define CK_MAILPROGRESS_API \
	void EmailReceived(const char *subject, \
			    const char *fromAddr, const char *fromName, \
			    const char *returnPath, \
			    const char *date, \
			    const char *uidl, \
			    int sizeInBytes);

#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

class CK_VISIBLE_PUBLIC CkMailManProgress  : public CkBaseProgress
{
    public:
	CkMailManProgress() { }
	virtual ~CkMailManProgress() { }

	virtual void EmailReceived(const char * /*subject*/, 
			    const char * /*fromAddr*/, 
			    const char * /*fromName*/, 
			    const char * /*returnPath*/, 
			    const char * /*date*/, 
			    const char * /*uidl*/, 
			    int /*sizeInBytes*/) { }


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


#endif
