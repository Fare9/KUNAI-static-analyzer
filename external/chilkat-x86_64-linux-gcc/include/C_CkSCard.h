// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkSCard_H
#define _C_CkSCard_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkSCard_setAbortCheck(HCkSCard cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkSCard_setPercentDone(HCkSCard cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkSCard_setProgressInfo(HCkSCard cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkSCard_setTaskCompleted(HCkSCard cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkSCard_setAbortCheck2(HCkSCard cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkSCard_setPercentDone2(HCkSCard cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkSCard_setProgressInfo2(HCkSCard cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkSCard_setTaskCompleted2(HCkSCard cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkSCard_setExternalProgress(HCkSCard cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkSCard_setCallbackContext(HCkSCard cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkSCard CkSCard_Create(void);
CK_C_VISIBLE_PUBLIC void CkSCard_Dispose(HCkSCard handle);
CK_C_VISIBLE_PUBLIC void CkSCard_getActiveProtocol(HCkSCard cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkSCard_activeProtocol(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC void CkSCard_getCardAtr(HCkSCard cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkSCard_cardAtr(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC void CkSCard_getConnectedReader(HCkSCard cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkSCard_connectedReader(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC void CkSCard_getContext(HCkSCard cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkSCard_context(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC void CkSCard_getDebugLogFilePath(HCkSCard cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkSCard_putDebugLogFilePath(HCkSCard cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkSCard_debugLogFilePath(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC void CkSCard_getLastErrorHtml(HCkSCard cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkSCard_lastErrorHtml(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC void CkSCard_getLastErrorText(HCkSCard cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkSCard_lastErrorText(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC void CkSCard_getLastErrorXml(HCkSCard cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkSCard_lastErrorXml(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_getLastMethodSuccess(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC void CkSCard_putLastMethodSuccess(HCkSCard cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkSCard_getPcscLibPath(HCkSCard cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkSCard_putPcscLibPath(HCkSCard cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkSCard_pcscLibPath(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC void CkSCard_getReaderStatus(HCkSCard cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkSCard_readerStatus(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC void CkSCard_getScardError(HCkSCard cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkSCard_scardError(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_getUtf8(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC void CkSCard_putUtf8(HCkSCard cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_getVerboseLogging(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC void CkSCard_putVerboseLogging(HCkSCard cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkSCard_getVersion(HCkSCard cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkSCard_version(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_BeginTransaction(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_CheckStatus(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_Connect(HCkSCard cHandle, const char *reader, const char *shareMode, const char *preferredProtocol);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_Disconnect(HCkSCard cHandle, const char *disposition);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_EndTransaction(HCkSCard cHandle, const char *disposition);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_EstablishContext(HCkSCard cHandle, const char *scope);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_FindSmartcards(HCkSCard cHandle, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_GetAttrib(HCkSCard cHandle, const char *attr, HCkBinData bd);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_GetAttribStr(HCkSCard cHandle, const char *attr, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkSCard_getAttribStr(HCkSCard cHandle, const char *attr);
CK_C_VISIBLE_PUBLIC unsigned long CkSCard_GetAttribUint(HCkSCard cHandle, const char *attr);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_GetStatusChange(HCkSCard cHandle, int maxWaitMs, HCkStringTable stReaderNames, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC HCkTask CkSCard_GetStatusChangeAsync(HCkSCard cHandle, int maxWaitMs, HCkStringTable stReaderNames, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_GetStatusChangeCancel(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_ListReaderGroups(HCkSCard cHandle, HCkStringTable readerGroups);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_ListReaders(HCkSCard cHandle, HCkStringTable st);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_Reconnect(HCkSCard cHandle, const char *shareMode, const char *preferredProtocol, const char *action);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_ReleaseContext(HCkSCard cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_SaveLastError(HCkSCard cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_SendControl(HCkSCard cHandle, unsigned long controlCode, HCkBinData bdSend, HCkBinData bdRecv);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_SendControlHex(HCkSCard cHandle, unsigned long controlCode, const char *sendData, HCkBinData bdRecv);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_Transmit(HCkSCard cHandle, const char *protocol, HCkBinData bdSend, HCkBinData bdRecv, int maxRecvLen);
CK_C_VISIBLE_PUBLIC BOOL CkSCard_TransmitHex(HCkSCard cHandle, const char *protocol, const char *apduHex, HCkBinData bdRecv, int maxRecvLen);
#endif
