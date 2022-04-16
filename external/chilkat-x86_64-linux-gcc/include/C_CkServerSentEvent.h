// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkServerSentEvent_H
#define _C_CkServerSentEvent_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkServerSentEvent CkServerSentEvent_Create(void);
CK_C_VISIBLE_PUBLIC void CkServerSentEvent_Dispose(HCkServerSentEvent handle);
CK_C_VISIBLE_PUBLIC void CkServerSentEvent_getData(HCkServerSentEvent cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkServerSentEvent_data(HCkServerSentEvent cHandle);
CK_C_VISIBLE_PUBLIC void CkServerSentEvent_getEventName(HCkServerSentEvent cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkServerSentEvent_eventName(HCkServerSentEvent cHandle);
CK_C_VISIBLE_PUBLIC void CkServerSentEvent_getLastEventId(HCkServerSentEvent cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkServerSentEvent_lastEventId(HCkServerSentEvent cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkServerSentEvent_getLastMethodSuccess(HCkServerSentEvent cHandle);
CK_C_VISIBLE_PUBLIC void CkServerSentEvent_putLastMethodSuccess(HCkServerSentEvent cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkServerSentEvent_getRetry(HCkServerSentEvent cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkServerSentEvent_getUtf8(HCkServerSentEvent cHandle);
CK_C_VISIBLE_PUBLIC void CkServerSentEvent_putUtf8(HCkServerSentEvent cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkServerSentEvent_LoadEvent(HCkServerSentEvent cHandle, const char *eventText);
#endif
