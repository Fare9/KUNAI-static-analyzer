// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkMailManWH
#define _C_CkMailManWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkMailManW_setAbortCheck(HCkMailManW cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkMailManW_setPercentDone(HCkMailManW cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkMailManW_setProgressInfo(HCkMailManW cHandle, void (*fnProgressInfo)(const wchar_t *name, const wchar_t *value));
CK_C_VISIBLE_PUBLIC void CkMailManW_setTaskCompleted(HCkMailManW cHandle, void (*fnTaskCompleted)(HCkTaskW hTask));

CK_C_VISIBLE_PUBLIC HCkMailManW CkMailManW_Create(void);
CK_C_VISIBLE_PUBLIC void CkMailManW_Dispose(HCkMailManW handle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getAbortCurrent(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putAbortCurrent(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getAllOrNone(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putAllOrNone(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getAutoFix(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putAutoFix(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getAutoGenMessageId(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putAutoGenMessageId(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getAutoSmtpRset(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putAutoSmtpRset(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getAutoUnwrapSecurity(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putAutoUnwrapSecurity(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getClientIpAddress(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putClientIpAddress(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_clientIpAddress(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC int CkMailManW_getConnectFailReason(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC int CkMailManW_getConnectTimeout(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putConnectTimeout(HCkMailManW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getDebugLogFilePath(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putDebugLogFilePath(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_debugLogFilePath(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getDsnEnvid(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putDsnEnvid(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_dsnEnvid(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getDsnNotify(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putDsnNotify(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_dsnNotify(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getDsnRet(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putDsnRet(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_dsnRet(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getEmbedCertChain(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putEmbedCertChain(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getFilter(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putFilter(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_filter(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC int CkMailManW_getHeartbeatMs(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putHeartbeatMs(HCkMailManW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getHeloHostname(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putHeloHostname(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_heloHostname(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getHttpProxyAuthMethod(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putHttpProxyAuthMethod(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_httpProxyAuthMethod(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getHttpProxyDomain(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putHttpProxyDomain(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_httpProxyDomain(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getHttpProxyHostname(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putHttpProxyHostname(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_httpProxyHostname(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getHttpProxyPassword(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putHttpProxyPassword(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_httpProxyPassword(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC int CkMailManW_getHttpProxyPort(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putHttpProxyPort(HCkMailManW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getHttpProxyUsername(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putHttpProxyUsername(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_httpProxyUsername(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getImmediateDelete(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putImmediateDelete(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getIncludeRootCert(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putIncludeRootCert(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getIsPop3Connected(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getIsSmtpConnected(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getLastErrorHtml(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_lastErrorHtml(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getLastErrorText(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_lastErrorText(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getLastErrorXml(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_lastErrorXml(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getLastMethodSuccess(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putLastMethodSuccess(HCkMailManW cHandle, BOOL newVal);
#if defined(CK_SMTPQ_INCLUDED)
CK_C_VISIBLE_PUBLIC void CkMailManW_getLastSendQFilename(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_lastSendQFilename(HCkMailManW cHandle);
#endif
CK_C_VISIBLE_PUBLIC int CkMailManW_getLastSmtpStatus(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getLastSmtpStatusMsg(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_lastSmtpStatusMsg(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getLogMailReceivedFilename(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putLogMailReceivedFilename(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_logMailReceivedFilename(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getLogMailSentFilename(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putLogMailSentFilename(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_logMailSentFilename(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getMailHost(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putMailHost(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_mailHost(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC int CkMailManW_getMailPort(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putMailPort(HCkMailManW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkMailManW_getMaxCount(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putMaxCount(HCkMailManW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getOAuth2AccessToken(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putOAuth2AccessToken(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_oAuth2AccessToken(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getOpaqueSigning(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putOpaqueSigning(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getP7mEncryptAttachFilename(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putP7mEncryptAttachFilename(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_p7mEncryptAttachFilename(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getP7mSigAttachFilename(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putP7mSigAttachFilename(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_p7mSigAttachFilename(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getP7sSigAttachFilename(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putP7sSigAttachFilename(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_p7sSigAttachFilename(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC int CkMailManW_getPercentDoneScale(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putPercentDoneScale(HCkMailManW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkMailManW_getPop3SessionId(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getPop3SessionLog(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_pop3SessionLog(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getPop3SPA(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putPop3SPA(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getPop3SslServerCertVerified(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getPop3Stls(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putPop3Stls(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getPopPassword(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putPopPassword(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_popPassword(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getPopPasswordBase64(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putPopPasswordBase64(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_popPasswordBase64(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getPopSsl(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putPopSsl(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getPopUsername(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putPopUsername(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_popUsername(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getPreferIpv6(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putPreferIpv6(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkMailManW_getReadTimeout(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putReadTimeout(HCkMailManW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getRequireSslCertVerify(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putRequireSslCertVerify(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getResetDateOnLoad(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putResetDateOnLoad(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkMailManW_getSendBufferSize(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSendBufferSize(HCkMailManW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getSendIndividual(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSendIndividual(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkMailManW_getSizeLimit(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSizeLimit(HCkMailManW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getSmtpAuthMethod(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSmtpAuthMethod(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_smtpAuthMethod(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getSmtpFailReason(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_smtpFailReason(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getSmtpHost(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSmtpHost(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_smtpHost(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getSmtpLoginDomain(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSmtpLoginDomain(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_smtpLoginDomain(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getSmtpPassword(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSmtpPassword(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_smtpPassword(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getSmtpPipelining(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSmtpPipelining(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkMailManW_getSmtpPort(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSmtpPort(HCkMailManW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getSmtpSessionLog(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_smtpSessionLog(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getSmtpSsl(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSmtpSsl(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getSmtpSslServerCertVerified(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getSmtpUsername(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSmtpUsername(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_smtpUsername(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getSocksHostname(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSocksHostname(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_socksHostname(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getSocksPassword(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSocksPassword(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_socksPassword(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC int CkMailManW_getSocksPort(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSocksPort(HCkMailManW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getSocksUsername(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSocksUsername(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_socksUsername(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC int CkMailManW_getSocksVersion(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSocksVersion(HCkMailManW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkMailManW_getSoRcvBuf(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSoRcvBuf(HCkMailManW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkMailManW_getSoSndBuf(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSoSndBuf(HCkMailManW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getSslAllowedCiphers(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSslAllowedCiphers(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_sslAllowedCiphers(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getSslProtocol(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putSslProtocol(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_sslProtocol(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getStartTLS(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putStartTLS(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getStartTLSifPossible(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putStartTLSifPossible(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getTlsCipherSuite(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_tlsCipherSuite(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getTlsPinSet(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putTlsPinSet(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_tlsPinSet(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getTlsVersion(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_tlsVersion(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_getUncommonOptions(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putUncommonOptions(HCkMailManW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_uncommonOptions(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getUseApop(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putUseApop(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_getVerboseLogging(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMailManW_putVerboseLogging(HCkMailManW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkMailManW_getVersion(HCkMailManW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_version(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_AddPfxSourceData(HCkMailManW cHandle, HCkByteData pfxData, const wchar_t *password);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_AddPfxSourceFile(HCkMailManW cHandle, const wchar_t *pfxFilePath, const wchar_t *password);
CK_C_VISIBLE_PUBLIC int CkMailManW_CheckMail(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_CheckMailAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_ClearBadEmailAddresses(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_ClearPop3SessionLog(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC void CkMailManW_ClearSmtpSessionLog(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_CloseSmtpConnection(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_CloseSmtpConnectionAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkEmailBundleW CkMailManW_CopyMail(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_CopyMailAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_DeleteBundle(HCkMailManW cHandle, HCkEmailBundleW emailBundle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_DeleteBundleAsync(HCkMailManW cHandle, HCkEmailBundleW emailBundle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_DeleteByMsgnum(HCkMailManW cHandle, int msgnum);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_DeleteByMsgnumAsync(HCkMailManW cHandle, int msgnum);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_DeleteByUidl(HCkMailManW cHandle, const wchar_t *uidl);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_DeleteByUidlAsync(HCkMailManW cHandle, const wchar_t *uidl);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_DeleteEmail(HCkMailManW cHandle, HCkEmailW email);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_DeleteEmailAsync(HCkMailManW cHandle, HCkEmailW email);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_DeleteMultiple(HCkMailManW cHandle, HCkStringArrayW uidlArray);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_DeleteMultipleAsync(HCkMailManW cHandle, HCkStringArrayW uidlArray);
CK_C_VISIBLE_PUBLIC HCkEmailW CkMailManW_FetchByMsgnum(HCkMailManW cHandle, int msgnum);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_FetchByMsgnumAsync(HCkMailManW cHandle, int msgnum);
CK_C_VISIBLE_PUBLIC HCkEmailW CkMailManW_FetchEmail(HCkMailManW cHandle, const wchar_t *uidl);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_FetchEmailAsync(HCkMailManW cHandle, const wchar_t *uidl);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_FetchMime(HCkMailManW cHandle, const wchar_t *uidl, HCkByteData outData);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_FetchMimeAsync(HCkMailManW cHandle, const wchar_t *uidl);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_FetchMimeBd(HCkMailManW cHandle, const wchar_t *uidl, HCkBinDataW mimeData);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_FetchMimeBdAsync(HCkMailManW cHandle, const wchar_t *uidl, HCkBinDataW mimeData);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_FetchMimeByMsgnum(HCkMailManW cHandle, int msgnum, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_FetchMimeByMsgnumAsync(HCkMailManW cHandle, int msgnum);
CK_C_VISIBLE_PUBLIC HCkEmailBundleW CkMailManW_FetchMultiple(HCkMailManW cHandle, HCkStringArrayW uidlArray);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_FetchMultipleAsync(HCkMailManW cHandle, HCkStringArrayW uidlArray);
CK_C_VISIBLE_PUBLIC HCkEmailBundleW CkMailManW_FetchMultipleHeaders(HCkMailManW cHandle, HCkStringArrayW uidlArray, int numBodyLines);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_FetchMultipleHeadersAsync(HCkMailManW cHandle, HCkStringArrayW uidlArray, int numBodyLines);
CK_C_VISIBLE_PUBLIC HCkStringArrayW CkMailManW_FetchMultipleMime(HCkMailManW cHandle, HCkStringArrayW uidlArray);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_FetchMultipleMimeAsync(HCkMailManW cHandle, HCkStringArrayW uidlArray);
CK_C_VISIBLE_PUBLIC HCkEmailW CkMailManW_FetchSingleHeader(HCkMailManW cHandle, int numBodyLines, int messageNumber);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_FetchSingleHeaderAsync(HCkMailManW cHandle, int numBodyLines, int messageNumber);
CK_C_VISIBLE_PUBLIC HCkEmailW CkMailManW_FetchSingleHeaderByUidl(HCkMailManW cHandle, int numBodyLines, const wchar_t *uidl);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_FetchSingleHeaderByUidlAsync(HCkMailManW cHandle, int numBodyLines, const wchar_t *uidl);
CK_C_VISIBLE_PUBLIC HCkEmailBundleW CkMailManW_GetAllHeaders(HCkMailManW cHandle, int numBodyLines);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_GetAllHeadersAsync(HCkMailManW cHandle, int numBodyLines);
CK_C_VISIBLE_PUBLIC HCkStringArrayW CkMailManW_GetBadEmailAddrs(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkEmailW CkMailManW_GetFullEmail(HCkMailManW cHandle, HCkEmailW email);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_GetFullEmailAsync(HCkMailManW cHandle, HCkEmailW email);
CK_C_VISIBLE_PUBLIC HCkEmailBundleW CkMailManW_GetHeaders(HCkMailManW cHandle, int numBodyLines, int fromIndex, int toIndex);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_GetHeadersAsync(HCkMailManW cHandle, int numBodyLines, int fromIndex, int toIndex);
CK_C_VISIBLE_PUBLIC int CkMailManW_GetMailboxCount(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_GetMailboxCountAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_GetMailboxInfoXml(HCkMailManW cHandle, HCkString outXml);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_getMailboxInfoXml(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_GetMailboxInfoXmlAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC unsigned long CkMailManW_GetMailboxSize(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_GetMailboxSizeAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkCertW CkMailManW_GetPop3SslServerCert(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkStringArrayW CkMailManW_GetSentToEmailAddrs(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC int CkMailManW_GetSizeByUidl(HCkMailManW cHandle, const wchar_t *uidl);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_GetSizeByUidlAsync(HCkMailManW cHandle, const wchar_t *uidl);
CK_C_VISIBLE_PUBLIC HCkCertW CkMailManW_GetSmtpSslServerCert(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkStringArrayW CkMailManW_GetUidls(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_GetUidlsAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_IsSmtpDsnCapable(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_IsSmtpDsnCapableAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_IsUnlocked(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkJsonObjectW CkMailManW_LastJsonData(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkEmailW CkMailManW_LoadEml(HCkMailManW cHandle, const wchar_t *emlFilename);
CK_C_VISIBLE_PUBLIC HCkEmailBundleW CkMailManW_LoadMbx(HCkMailManW cHandle, const wchar_t *mbxFileName);
CK_C_VISIBLE_PUBLIC HCkEmailW CkMailManW_LoadMime(HCkMailManW cHandle, const wchar_t *mimeText);
#if defined(CK_SMTPQ_INCLUDED)
CK_C_VISIBLE_PUBLIC HCkEmailW CkMailManW_LoadQueuedEmail(HCkMailManW cHandle, const wchar_t *path);
#endif
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_LoadTaskCaller(HCkMailManW cHandle, HCkTaskW task);
CK_C_VISIBLE_PUBLIC HCkEmailW CkMailManW_LoadXmlEmail(HCkMailManW cHandle, const wchar_t *filename);
CK_C_VISIBLE_PUBLIC HCkEmailW CkMailManW_LoadXmlEmailString(HCkMailManW cHandle, const wchar_t *xmlString);
CK_C_VISIBLE_PUBLIC HCkEmailBundleW CkMailManW_LoadXmlFile(HCkMailManW cHandle, const wchar_t *filename);
CK_C_VISIBLE_PUBLIC HCkEmailBundleW CkMailManW_LoadXmlString(HCkMailManW cHandle, const wchar_t *xmlString);
#if defined(CK_MX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_MxLookup(HCkMailManW cHandle, const wchar_t *emailAddress, HCkString outStrHostname);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_mxLookup(HCkMailManW cHandle, const wchar_t *emailAddress);
#endif
#if defined(CK_MX_INCLUDED)
CK_C_VISIBLE_PUBLIC HCkStringArrayW CkMailManW_MxLookupAll(HCkMailManW cHandle, const wchar_t *emailAddress);
#endif
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_OpenSmtpConnection(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_OpenSmtpConnectionAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_Pop3Authenticate(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_Pop3AuthenticateAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_Pop3BeginSession(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_Pop3BeginSessionAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_Pop3Connect(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_Pop3ConnectAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_Pop3EndSession(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_Pop3EndSessionAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_Pop3EndSessionNoQuit(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_Pop3EndSessionNoQuitAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_Pop3Noop(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_Pop3NoopAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_Pop3Reset(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_Pop3ResetAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_Pop3SendRawCommand(HCkMailManW cHandle, const wchar_t *command, const wchar_t *charset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_pop3SendRawCommand(HCkMailManW cHandle, const wchar_t *command, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_Pop3SendRawCommandAsync(HCkMailManW cHandle, const wchar_t *command, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_QuickSend(HCkMailManW cHandle, const wchar_t *fromAddr, const wchar_t *toAddr, const wchar_t *subject, const wchar_t *body, const wchar_t *smtpServer);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_QuickSendAsync(HCkMailManW cHandle, const wchar_t *fromAddr, const wchar_t *toAddr, const wchar_t *subject, const wchar_t *body, const wchar_t *smtpServer);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_RenderToMime(HCkMailManW cHandle, HCkEmailW email, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_renderToMime(HCkMailManW cHandle, HCkEmailW email);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_RenderToMimeBd(HCkMailManW cHandle, HCkEmailW email, HCkBinDataW renderedMime);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_RenderToMimeBytes(HCkMailManW cHandle, HCkEmailW email, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_RenderToMimeSb(HCkMailManW cHandle, HCkEmailW email, HCkStringBuilderW renderedMime);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SaveLastError(HCkMailManW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SendBundle(HCkMailManW cHandle, HCkEmailBundleW bundle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SendBundleAsync(HCkMailManW cHandle, HCkEmailBundleW bundle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SendEmail(HCkMailManW cHandle, HCkEmailW email);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SendEmailAsync(HCkMailManW cHandle, HCkEmailW email);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SendMime(HCkMailManW cHandle, const wchar_t *fromAddr, const wchar_t *recipients, const wchar_t *mimeSource);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SendMimeAsync(HCkMailManW cHandle, const wchar_t *fromAddr, const wchar_t *recipients, const wchar_t *mimeSource);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SendMimeBd(HCkMailManW cHandle, const wchar_t *fromAddr, const wchar_t *recipients, HCkBinDataW mimeData);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SendMimeBdAsync(HCkMailManW cHandle, const wchar_t *fromAddr, const wchar_t *recipients, HCkBinDataW mimeData);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SendMimeBytes(HCkMailManW cHandle, const wchar_t *fromAddr, const wchar_t *recipients, HCkByteData mimeSource);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SendMimeBytesAsync(HCkMailManW cHandle, const wchar_t *fromAddr, const wchar_t *recipients, HCkByteData mimeSource);
#if defined(CK_SMTPQ_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SendMimeBytesQ(HCkMailManW cHandle, const wchar_t *from, const wchar_t *recipients, HCkByteData mimeData);
#endif
#if defined(CK_SMTPQ_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SendMimeQ(HCkMailManW cHandle, const wchar_t *fromAddr, const wchar_t *recipients, const wchar_t *mimeSource);
#endif
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SendMimeToList(HCkMailManW cHandle, const wchar_t *fromAddr, const wchar_t *distListFilename, const wchar_t *mimeSource);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SendMimeToListAsync(HCkMailManW cHandle, const wchar_t *fromAddr, const wchar_t *distListFilename, const wchar_t *mimeSource);
#if defined(CK_SMTPQ_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SendQ(HCkMailManW cHandle, HCkEmailW email);
#endif
#if defined(CK_SMTPQ_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SendQ2(HCkMailManW cHandle, HCkEmailW email, const wchar_t *queueDir);
#endif
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SendToDistributionList(HCkMailManW cHandle, HCkEmailW emailObj, HCkStringArrayW recipientList);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SendToDistributionListAsync(HCkMailManW cHandle, HCkEmailW emailObj, HCkStringArrayW recipientList);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SetDecryptCert(HCkMailManW cHandle, HCkCertW cert);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SetDecryptCert2(HCkMailManW cHandle, HCkCertW cert, HCkPrivateKeyW privateKey);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SetPassword(HCkMailManW cHandle, const wchar_t *protocol, HCkSecureStringW password);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SetSslClientCert(HCkMailManW cHandle, HCkCertW cert);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SetSslClientCertPem(HCkMailManW cHandle, const wchar_t *pemDataOrFilename, const wchar_t *pemPassword);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SetSslClientCertPfx(HCkMailManW cHandle, const wchar_t *pfxFilename, const wchar_t *pfxPassword);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SmtpAuthenticate(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SmtpAuthenticateAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SmtpConnect(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SmtpConnectAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SmtpNoop(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SmtpNoopAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SmtpReset(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SmtpResetAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SmtpSendRawCommand(HCkMailManW cHandle, const wchar_t *command, const wchar_t *charset, BOOL bEncodeBase64, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMailManW_smtpSendRawCommand(HCkMailManW cHandle, const wchar_t *command, const wchar_t *charset, BOOL bEncodeBase64);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SmtpSendRawCommandAsync(HCkMailManW cHandle, const wchar_t *command, const wchar_t *charset, BOOL bEncodeBase64);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SshAuthenticatePk(HCkMailManW cHandle, const wchar_t *sshLogin, HCkSshKeyW sshUsername);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SshAuthenticatePkAsync(HCkMailManW cHandle, const wchar_t *sshLogin, HCkSshKeyW sshUsername);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SshAuthenticatePw(HCkMailManW cHandle, const wchar_t *sshLogin, const wchar_t *sshPassword);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SshAuthenticatePwAsync(HCkMailManW cHandle, const wchar_t *sshLogin, const wchar_t *sshPassword);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SshCloseTunnel(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SshCloseTunnelAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_SshOpenTunnel(HCkMailManW cHandle, const wchar_t *sshHostname, int sshPort);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_SshOpenTunnelAsync(HCkMailManW cHandle, const wchar_t *sshHostname, int sshPort);
CK_C_VISIBLE_PUBLIC HCkEmailBundleW CkMailManW_TransferMail(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_TransferMailAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkStringArrayW CkMailManW_TransferMultipleMime(HCkMailManW cHandle, HCkStringArrayW uidlArray);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_TransferMultipleMimeAsync(HCkMailManW cHandle, HCkStringArrayW uidlArray);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_UnlockComponent(HCkMailManW cHandle, const wchar_t *code);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_UseCertVault(HCkMailManW cHandle, HCkXmlCertVaultW vault);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_UseSsh(HCkMailManW cHandle, HCkSshW ssh);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_UseSshTunnel(HCkMailManW cHandle, HCkSocketW tunnel);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_VerifyPopConnection(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_VerifyPopConnectionAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_VerifyPopLogin(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_VerifyPopLoginAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_VerifyRecips(HCkMailManW cHandle, HCkEmailW email, HCkStringArrayW badAddrs);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_VerifyRecipsAsync(HCkMailManW cHandle, HCkEmailW email, HCkStringArrayW badAddrs);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_VerifySmtpConnection(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_VerifySmtpConnectionAsync(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailManW_VerifySmtpLogin(HCkMailManW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkMailManW_VerifySmtpLoginAsync(HCkMailManW cHandle);
#endif
