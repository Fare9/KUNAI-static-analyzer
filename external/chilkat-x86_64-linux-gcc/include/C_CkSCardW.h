// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkSCardWH
#define _C_CkSCardWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkSCardW_setAbortCheck(HCkSCardW cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkSCardW_setPercentDone(HCkSCardW cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkSCardW_setProgressInfo(HCkSCardW cHandle, void (*fnProgressInfo)(const wchar_t *name, const wchar_t *value));
CK_C_VISIBLE_PUBLIC void CkSCardW_setTaskCompleted(HCkSCardW cHandle, void (*fnTaskCompleted)(HCkTaskW hTask));

CK_C_VISIBLE_PUBLIC HCkSCardW CkSCardW_Create(void);
CK_C_VISIBLE_PUBLIC void CkSCardW_Dispose(HCkSCardW handle);
CK_C_VISIBLE_PUBLIC void CkSCardW_getActiveProtocol(HCkSCardW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSCardW_activeProtocol(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC void CkSCardW_getCardAtr(HCkSCardW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSCardW_cardAtr(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC void CkSCardW_getConnectedReader(HCkSCardW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSCardW_connectedReader(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC void CkSCardW_getContext(HCkSCardW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSCardW_context(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC void CkSCardW_getDebugLogFilePath(HCkSCardW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkSCardW_putDebugLogFilePath(HCkSCardW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSCardW_debugLogFilePath(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC void CkSCardW_getLastErrorHtml(HCkSCardW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSCardW_lastErrorHtml(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC void CkSCardW_getLastErrorText(HCkSCardW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSCardW_lastErrorText(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC void CkSCardW_getLastErrorXml(HCkSCardW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSCardW_lastErrorXml(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_getLastMethodSuccess(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC void  CkSCardW_putLastMethodSuccess(HCkSCardW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkSCardW_getPcscLibPath(HCkSCardW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkSCardW_putPcscLibPath(HCkSCardW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSCardW_pcscLibPath(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC void CkSCardW_getReaderStatus(HCkSCardW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSCardW_readerStatus(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC void CkSCardW_getScardError(HCkSCardW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSCardW_scardError(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_getVerboseLogging(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC void  CkSCardW_putVerboseLogging(HCkSCardW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkSCardW_getVersion(HCkSCardW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSCardW_version(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_BeginTransaction(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_CheckStatus(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_Connect(HCkSCardW cHandle, const wchar_t *reader, const wchar_t *shareMode, const wchar_t *preferredProtocol);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_Disconnect(HCkSCardW cHandle, const wchar_t *disposition);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_EndTransaction(HCkSCardW cHandle, const wchar_t *disposition);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_EstablishContext(HCkSCardW cHandle, const wchar_t *scope);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_FindSmartcards(HCkSCardW cHandle, HCkJsonObjectW json);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_GetAttrib(HCkSCardW cHandle, const wchar_t *attr, HCkBinDataW bd);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_GetAttribStr(HCkSCardW cHandle, const wchar_t *attr, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSCardW_getAttribStr(HCkSCardW cHandle, const wchar_t *attr);
CK_C_VISIBLE_PUBLIC unsigned long CkSCardW_GetAttribUint(HCkSCardW cHandle, const wchar_t *attr);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_GetStatusChange(HCkSCardW cHandle, int maxWaitMs, HCkStringTableW stReaderNames, HCkJsonObjectW json);
CK_C_VISIBLE_PUBLIC HCkTaskW CkSCardW_GetStatusChangeAsync(HCkSCardW cHandle, int maxWaitMs, HCkStringTableW stReaderNames, HCkJsonObjectW json);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_GetStatusChangeCancel(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_ListReaderGroups(HCkSCardW cHandle, HCkStringTableW readerGroups);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_ListReaders(HCkSCardW cHandle, HCkStringTableW st);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_Reconnect(HCkSCardW cHandle, const wchar_t *shareMode, const wchar_t *preferredProtocol, const wchar_t *action);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_ReleaseContext(HCkSCardW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_SaveLastError(HCkSCardW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_SendControl(HCkSCardW cHandle, unsigned long controlCode, HCkBinDataW bdSend, HCkBinDataW bdRecv);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_SendControlHex(HCkSCardW cHandle, unsigned long controlCode, const wchar_t *sendData, HCkBinDataW bdRecv);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_Transmit(HCkSCardW cHandle, const wchar_t *protocol, HCkBinDataW bdSend, HCkBinDataW bdRecv, int maxRecvLen);
CK_C_VISIBLE_PUBLIC BOOL CkSCardW_TransmitHex(HCkSCardW cHandle, const wchar_t *protocol, const wchar_t *apduHex, HCkBinDataW bdRecv, int maxRecvLen);
#endif
