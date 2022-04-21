// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkServerSentEventWH
#define _C_CkServerSentEventWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkServerSentEventW CkServerSentEventW_Create(void);
CK_C_VISIBLE_PUBLIC void CkServerSentEventW_Dispose(HCkServerSentEventW handle);
CK_C_VISIBLE_PUBLIC void CkServerSentEventW_getData(HCkServerSentEventW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkServerSentEventW_data(HCkServerSentEventW cHandle);
CK_C_VISIBLE_PUBLIC void CkServerSentEventW_getEventName(HCkServerSentEventW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkServerSentEventW_eventName(HCkServerSentEventW cHandle);
CK_C_VISIBLE_PUBLIC void CkServerSentEventW_getLastEventId(HCkServerSentEventW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkServerSentEventW_lastEventId(HCkServerSentEventW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkServerSentEventW_getLastMethodSuccess(HCkServerSentEventW cHandle);
CK_C_VISIBLE_PUBLIC void  CkServerSentEventW_putLastMethodSuccess(HCkServerSentEventW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkServerSentEventW_getRetry(HCkServerSentEventW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkServerSentEventW_LoadEvent(HCkServerSentEventW cHandle, const wchar_t *eventText);
#endif
