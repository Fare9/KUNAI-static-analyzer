// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkKeyContainerWH
#define _C_CkKeyContainerWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkKeyContainerW CkKeyContainerW_Create(void);
CK_C_VISIBLE_PUBLIC void CkKeyContainerW_Dispose(HCkKeyContainerW handle);
CK_C_VISIBLE_PUBLIC void CkKeyContainerW_getContainerName(HCkKeyContainerW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkKeyContainerW_containerName(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainerW_getDebugLogFilePath(HCkKeyContainerW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkKeyContainerW_putDebugLogFilePath(HCkKeyContainerW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkKeyContainerW_debugLogFilePath(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_getIsMachineKeyset(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_getIsOpen(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainerW_getLastErrorHtml(HCkKeyContainerW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkKeyContainerW_lastErrorHtml(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainerW_getLastErrorText(HCkKeyContainerW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkKeyContainerW_lastErrorText(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainerW_getLastErrorXml(HCkKeyContainerW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkKeyContainerW_lastErrorXml(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_getLastMethodSuccess(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC void  CkKeyContainerW_putLastMethodSuccess(HCkKeyContainerW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkKeyContainerW_getLegacyKeySpec(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC void  CkKeyContainerW_putLegacyKeySpec(HCkKeyContainerW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_getMachineKeys(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC void  CkKeyContainerW_putMachineKeys(HCkKeyContainerW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_getVerboseLogging(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC void  CkKeyContainerW_putVerboseLogging(HCkKeyContainerW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkKeyContainerW_getVersion(HCkKeyContainerW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkKeyContainerW_version(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainerW_CloseContainer(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_CreateContainer(HCkKeyContainerW cHandle, const wchar_t *name, BOOL machineKeyset);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_DeleteContainer(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_DeleteKey(HCkKeyContainerW cHandle, const wchar_t *keyName, const wchar_t *storageProvider);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_ExportKey(HCkKeyContainerW cHandle, const wchar_t *keyName, const wchar_t *storageProvider, BOOL silentFlag, HCkPrivateKeyW key);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_FetchContainerNames(HCkKeyContainerW cHandle, BOOL bMachineKeyset);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_GenerateKeyPair(HCkKeyContainerW cHandle, BOOL bKeyExchangePair, int keyLengthInBits);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_GenerateUuid(HCkKeyContainerW cHandle, HCkString outGuid);
CK_C_VISIBLE_PUBLIC const wchar_t *CkKeyContainerW_generateUuid(HCkKeyContainerW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_GetKeys(HCkKeyContainerW cHandle, const wchar_t *storageProvider, HCkJsonObjectW json);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_GetNthContainerName(HCkKeyContainerW cHandle, BOOL bMachineKeyset, int index, HCkString outName);
CK_C_VISIBLE_PUBLIC const wchar_t *CkKeyContainerW_getNthContainerName(HCkKeyContainerW cHandle, BOOL bMachineKeyset, int index);
CK_C_VISIBLE_PUBLIC int CkKeyContainerW_GetNumContainers(HCkKeyContainerW cHandle, BOOL bMachineKeyset);
CK_C_VISIBLE_PUBLIC HCkPrivateKeyW CkKeyContainerW_GetPrivateKey(HCkKeyContainerW cHandle, BOOL bKeyExchangePair);
CK_C_VISIBLE_PUBLIC HCkPublicKeyW CkKeyContainerW_GetPublicKey(HCkKeyContainerW cHandle, BOOL bKeyExchangePair);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_GetStorageProviders(HCkKeyContainerW cHandle, HCkJsonObjectW json);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_ImportKey(HCkKeyContainerW cHandle, const wchar_t *keyName, const wchar_t *storageProvider, BOOL allowOverwrite, BOOL allowExport, HCkPrivateKeyW key);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_ImportPrivateKey(HCkKeyContainerW cHandle, HCkPrivateKeyW key, BOOL bKeyExchangePair);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_ImportPublicKey(HCkKeyContainerW cHandle, HCkPublicKeyW key, BOOL bKeyExchangePair);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_OpenContainer(HCkKeyContainerW cHandle, const wchar_t *name, BOOL needPrivateKeyAccess, BOOL machineKeyset);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainerW_SaveLastError(HCkKeyContainerW cHandle, const wchar_t *path);
#endif
