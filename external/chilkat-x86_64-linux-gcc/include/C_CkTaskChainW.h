// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkTaskChainWH
#define _C_CkTaskChainWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkTaskChainW CkTaskChainW_Create(void);
CK_C_VISIBLE_PUBLIC void CkTaskChainW_Dispose(HCkTaskChainW handle);
CK_C_VISIBLE_PUBLIC void CkTaskChainW_getDebugLogFilePath(HCkTaskChainW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkTaskChainW_putDebugLogFilePath(HCkTaskChainW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkTaskChainW_debugLogFilePath(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTaskChainW_getFinished(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC int CkTaskChainW_getHeartbeatMs(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC void  CkTaskChainW_putHeartbeatMs(HCkTaskChainW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkTaskChainW_getInert(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC void CkTaskChainW_getLastErrorHtml(HCkTaskChainW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkTaskChainW_lastErrorHtml(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC void CkTaskChainW_getLastErrorText(HCkTaskChainW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkTaskChainW_lastErrorText(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC void CkTaskChainW_getLastErrorXml(HCkTaskChainW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkTaskChainW_lastErrorXml(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTaskChainW_getLastMethodSuccess(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC void  CkTaskChainW_putLastMethodSuccess(HCkTaskChainW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkTaskChainW_getLive(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC int CkTaskChainW_getNumTasks(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC void CkTaskChainW_getStatus(HCkTaskChainW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkTaskChainW_status(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC int CkTaskChainW_getStatusInt(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTaskChainW_getStopOnFailedTask(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC void  CkTaskChainW_putStopOnFailedTask(HCkTaskChainW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkTaskChainW_getVerboseLogging(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC void  CkTaskChainW_putVerboseLogging(HCkTaskChainW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkTaskChainW_getVersion(HCkTaskChainW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkTaskChainW_version(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTaskChainW_Append(HCkTaskChainW cHandle, HCkTaskW task);
CK_C_VISIBLE_PUBLIC BOOL CkTaskChainW_Cancel(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkTaskChainW_GetTask(HCkTaskChainW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkTaskChainW_Run(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTaskChainW_RunSynchronously(HCkTaskChainW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTaskChainW_SaveLastError(HCkTaskChainW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC void CkTaskChainW_SleepMs(HCkTaskChainW cHandle, int numMs);
CK_C_VISIBLE_PUBLIC BOOL CkTaskChainW_Wait(HCkTaskChainW cHandle, int maxWaitMs);
#endif
