// generic p-code virtual machine
class PCodeVM {
    public float a;
    public int b;
    public int c;
    public long d;
    public long e;
    public double f;
    public Object g;
    public float h;
    public Object i;
    public double j;

    public int intStored;
    public int intLoaded;
    public Object objStored;
    public Object objLoaded;
    


    private final long[] jstk;
    private int stkidx;
    private final int[] istk;
    private final float[] fstk;
    private int peekidx;
    private final Object[] lstk;
    private final double[] dstk;

    public PCodeVM(Object arg3, Object arg4) {
        this.istk = new int[12];
        this.jstk = new long[12];
        this.fstk = new float[12];
        this.dstk = new double[12];
        this.lstk = new Object[12];
        this.lstk[5] = arg3;
        this.lstk[6] = arg4;
        this.stkidx = 0;
        this.peekidx = -1;
        this.intStored = 0;
        this.intLoaded = 0;
        this.objStored = null;
        this.objLoaded = null;
    }

    // returns 0 on completed execution, else, the input opcode
    public int exec(int opcode) {
        int v0 = 1;
        switch (opcode) {
            case 1: {
                int v2 = this.stkidx;
                this.stkidx = v2 + 1;
                this.lstk[v2] = this.lstk[6];
                int v3 = this.stkidx - 1;
                Object v0_1 = this.lstk[this.stkidx - 1];
                this.lstk[this.stkidx - 1] = null;
                this.istk[v3] = ((byte[]) v0_1).length;
                this.lstk[this.stkidx - 1] = new byte[this.istk[this.stkidx - 1]];
                return 0;
            }
            case 2: {
                --this.stkidx;
                Object v3_1 = this.lstk[this.stkidx];
                this.lstk[this.stkidx] = null;
                this.lstk[7] = v3_1;
                return 0;
            }
            case 3: {
                int v2_1 = this.stkidx;
                this.stkidx = v2_1 + 1;
                this.istk[v2_1] = 0;
                --this.stkidx;
                this.istk[8] = this.istk[this.stkidx];
                int v2_2 = this.stkidx;
                this.stkidx = v2_2 + 1;
                this.istk[v2_2] = 0;
                return 0;
            }
            case 4: {
                --this.stkidx;
                this.istk[9] = this.istk[this.stkidx];
                return 0;
            }
            case 5: {
                int v2_3 = this.stkidx;
                this.stkidx = v2_3 + 1;
                this.istk[v2_3] = 0;
                --this.stkidx;
                this.istk[11] = this.istk[this.stkidx];
                return 0;
            }
            case 6: {
                Object v0_2 = this.lstk[this.stkidx - 1];
                this.lstk[this.stkidx - 1] = null;
                this.i = v0_2;
                return 0;
            }
            case 7: {
                int v2_4 = this.stkidx;
                this.stkidx = v2_4 + 1;
                this.lstk[v2_4] = this.lstk[7];
                return 0;
            }
            case 8: {
                int v2_5 = this.stkidx;
                this.stkidx = v2_5 + 1;
                this.istk[v2_5] = this.c;
                return 0;
            }
            case 9: {
                int v2_6 = this.stkidx;
                this.stkidx = v2_6 + 1;
                this.istk[v2_6] = 43;
                --this.stkidx;
                this.istk[this.stkidx - 1] += this.istk[this.stkidx];
                int v2_7 = this.stkidx;
                this.stkidx = v2_7 + 1;
                this.istk[v2_7] = this.istk[this.stkidx - 2];
                return 0;
            }
            case 10: {
                int v2_8 = this.stkidx;
                this.stkidx = v2_8 + 1;
                this.istk[v2_8] = 0x80;
                return 0;
            }
            case 11: {
                int v0_3 = this.stkidx - this.c;
                this.stkidx = v0_3;
                this.peekidx = v0_3;
                return 0;
            }
            case 12: {
                int v2_9 = this.peekidx;
                this.peekidx = v2_9 + 1;
                this.b = this.istk[v2_9];
                return 0;
            }
            case 13: {
                --this.stkidx;
                this.istk[this.stkidx - 1] %= this.istk[this.stkidx];
                return 0;
            }
            case 14: {
                int v2_10 = this.stkidx;
                this.stkidx = v2_10 + 1;
                this.istk[v2_10] = 2;
                return 0;
            }
            case 15: {
                --this.stkidx;
                if (this.istk[this.stkidx] != 0) {
                    v0 = 0;
                }

                this.b = v0;
                return 0;
            }
            case 16: {
                int v2_11 = this.stkidx;
                this.stkidx = v2_11 + 1;
                this.istk[v2_11] = 0x5F;
                return 0;
            }
            case 17: {
                int v2_12 = this.stkidx;
                this.stkidx = v2_12 + 1;
                this.istk[v2_12] = 98;
                return 0;
            }
            case 18: {
                int v2_13 = this.stkidx;
                this.stkidx = v2_13 + 1;
                this.istk[v2_13] = 0;
                return 0;
            }
            case 19: {
                ++this.istk[10];
                return 0;
            }
            case 20: {
                int v2_14 = this.stkidx;
                this.stkidx = v2_14 + 1;
                this.lstk[v2_14] = this.lstk[7];
                int v2_15 = this.stkidx;
                this.stkidx = v2_15 + 1;
                this.istk[v2_15] = this.istk[11];
                return 0;
            }
            case 21: {
                int v2_16 = this.stkidx;
                this.stkidx = v2_16 + 1;
                this.lstk[v2_16] = this.lstk[6];
                int v2_17 = this.stkidx;
                this.stkidx = v2_17 + 1;
                this.istk[v2_17] = this.istk[11];
                --this.stkidx;
                int v3_2 = this.stkidx - 1;
                Object v0_4 = this.lstk[this.stkidx - 1];
                this.lstk[this.stkidx - 1] = null;
                this.istk[v3_2] = ((byte[]) v0_4)[this.istk[this.stkidx]];
                return 0;
            }
            case 22: {
                int v2_18 = this.stkidx;
                this.stkidx = v2_18 + 1;
                this.istk[v2_18] = this.istk[10];
                return 0;
            }
            case 23: {
                int v2_19 = this.stkidx;
                this.stkidx = v2_19 + 1;
                this.istk[v2_19] = 3;
                --this.stkidx;
                this.istk[this.stkidx - 1] += this.istk[this.stkidx];
                --this.stkidx;
                this.istk[this.stkidx - 1] ^= this.istk[this.stkidx];
                return 0;
            }
            case 24: {
                this.istk[this.stkidx - 1] = (byte) this.istk[this.stkidx - 1];
                return 0;
            }
            case 25: {
                this.stkidx += -3;
                Object v0_5 = this.lstk[this.stkidx];
                this.lstk[this.stkidx] = null;
                ((byte[]) v0_5)[this.istk[this.stkidx + 1]] = (byte)this.istk[this.stkidx + 2];
                ++this.istk[11];
                return 0;
            }
            case 26: {
                int v3_3 = this.stkidx;
                this.stkidx = v3_3 + 1;
                this.istk[v3_3] = 1;
                return 0;
            }
            case 27: {
                int v2_20 = this.stkidx;
                this.stkidx = v2_20 + 1;
                this.istk[v2_20] = 0x1F;
                return 0;
            }
            case 28: {
                --this.stkidx;
                this.istk[this.stkidx - 1] += this.istk[this.stkidx];
                int v2_21 = this.stkidx;
                this.stkidx = v2_21 + 1;
                this.istk[v2_21] = this.istk[this.stkidx - 2];
                int v2_22 = this.stkidx;
                this.stkidx = v2_22 + 1;
                this.istk[v2_22] = 0x80;
                return 0;
            }
            case 29: {
                --this.stkidx;
                if (this.istk[this.stkidx] == 0) {
                    v0 = 0;
                }

                this.b = v0;
                return 0;
            }
            case 30: {
                int v2_23 = this.stkidx - 1;
                this.stkidx = v2_23;
                this.b = this.istk[v2_23];
                return 0;
            }
            case 31: {
                --this.stkidx;
                this.lstk[this.stkidx] = null;
                return 0;
            }
            case 32: {
                int v3_4 = this.stkidx;
                this.stkidx = v3_4 + 1;
                this.istk[v3_4] = this.istk[8];
                int v3_5 = this.stkidx;
                this.stkidx = v3_5 + 1;
                this.istk[v3_5] = 1;
                --this.stkidx;
                this.istk[this.stkidx - 1] += this.istk[this.stkidx];
                return 0;
            }
            case 33: {
                int v2_24 = this.stkidx;
                this.stkidx = v2_24 + 1;
                this.istk[v2_24] = 0xFF;
                --this.stkidx;
                this.istk[this.stkidx - 1] &= this.istk[this.stkidx];
                return 0;
            }
            case 34: {
                --this.stkidx;
                this.istk[8] = this.istk[this.stkidx];
                return 0;
            }
            case 35: {
                int v2_25 = this.stkidx;
                this.stkidx = v2_25 + 1;
                this.istk[v2_25] = this.istk[9];
                return 0;
            }
            case 36: {
                int v2_26 = this.stkidx;
                this.stkidx = v2_26 + 1;
                this.lstk[v2_26] = this.lstk[5];
                return 0;
            }
            case 37: {
                int v2_27 = this.peekidx;
                this.peekidx = v2_27 + 1;
                Object v3_6 = this.lstk[v2_27];
                this.lstk[v2_27] = null;
                this.i = v3_6;
                return 0;
            }
            case 38: {
                int v2_28 = this.stkidx;
                this.stkidx = v2_28 + 1;
                this.lstk[v2_28] = this.g;
                return 0;
            }
            case 39: {
                int v2_29 = this.stkidx;
                this.stkidx = v2_29 + 1;
                this.istk[v2_29] = this.istk[8];
                --this.stkidx;
                int v3_7 = this.stkidx - 1;
                Object v0_6 = this.lstk[this.stkidx - 1];
                this.lstk[this.stkidx - 1] = null;
                this.istk[v3_7] = ((byte[]) v0_6)[this.istk[this.stkidx]];
                return 0;
            }
            case 40: {
                --this.stkidx;
                this.istk[this.stkidx - 1] += this.istk[this.stkidx];
                return 0;
            }
            case 41: {
                int v2_30 = this.stkidx;
                this.stkidx = v2_30 + 1;
                this.istk[v2_30] = 0xFF;
                return 0;
            }
            case 42: {
                --this.stkidx;
                this.istk[this.stkidx - 1] &= this.istk[this.stkidx];
                return 0;
            }
            case 43: {
                --this.stkidx;
                this.istk[9] = this.istk[this.stkidx];
                int v2_31 = this.stkidx;
                this.stkidx = v2_31 + 1;
                this.lstk[v2_31] = this.lstk[5];
                return 0;
            }
            case 44: {
                int v2_32 = this.stkidx;
                this.stkidx = v2_32 + 1;
                this.istk[v2_32] = this.istk[9];
                --this.stkidx;
                int v3_8 = this.stkidx - 1;
                Object v0_7 = this.lstk[this.stkidx - 1];
                this.lstk[this.stkidx - 1] = null;
                this.istk[v3_8] = ((byte[]) v0_7)[this.istk[this.stkidx]];
                --this.stkidx;
                this.istk[10] = this.istk[this.stkidx];
                return 0;
            }
            case 45: {
                int v2_33 = this.stkidx;
                this.stkidx = v2_33 + 1;
                Object v3_9 = this.lstk[this.stkidx - 2];
                this.lstk[this.stkidx - 2] = null;
                this.lstk[v2_33] = v3_9;
                this.istk[this.stkidx - 2] = this.istk[this.stkidx - 3];
                this.lstk[this.stkidx - 3] = v3_9;
                return 0;
            }
            case 46: {
                int v2_34 = this.stkidx;
                this.stkidx = v2_34 + 1;
                this.istk[v2_34] = this.istk[8];
                return 0;
            }
            case 47: {
                --this.stkidx;
                int v3_10 = this.stkidx - 1;
                Object v0_8 = this.lstk[this.stkidx - 1];
                this.lstk[this.stkidx - 1] = null;
                this.istk[v3_10] = ((byte[]) v0_8)[this.istk[this.stkidx]];
                this.stkidx += -3;
                Object v0_9 = this.lstk[this.stkidx];
                this.lstk[this.stkidx] = null;
                ((byte[]) v0_9)[this.istk[this.stkidx + 1]] = (byte)this.istk[this.stkidx + 2];
                return 0;
            }
            case 48: {
                int v2_35 = this.stkidx;
                this.stkidx = v2_35 + 1;
                this.istk[v2_35] = this.istk[10];
                this.stkidx += -3;
                Object v0_10 = this.lstk[this.stkidx];
                this.lstk[this.stkidx] = null;
                ((byte[]) v0_10)[this.istk[this.stkidx + 1]] = (byte)this.istk[this.stkidx + 2];
                return 0;
            }
            case 49: {
                --this.stkidx;
                int v3_11 = this.stkidx - 1;
                Object v0_11 = this.lstk[this.stkidx - 1];
                this.lstk[this.stkidx - 1] = null;
                this.istk[v3_11] = ((byte[]) v0_11)[this.istk[this.stkidx]];
                int v2_36 = this.stkidx;
                this.stkidx = v2_36 + 1;
                this.lstk[v2_36] = this.lstk[5];
                return 0;
            }
            case 50: {
                int v2_37 = this.stkidx;
                this.stkidx = v2_37 + 1;
                this.istk[v2_37] = this.istk[9];
                --this.stkidx;
                int v3_12 = this.stkidx - 1;
                Object v0_12 = this.lstk[this.stkidx - 1];
                this.lstk[this.stkidx - 1] = null;
                this.istk[v3_12] = ((byte[]) v0_12)[this.istk[this.stkidx]];
                --this.stkidx;
                this.istk[this.stkidx - 1] += this.istk[this.stkidx];
                return 0;
            }
            case 51: {
                --this.stkidx;
                this.istk[this.stkidx - 1] &= this.istk[this.stkidx];
                --this.stkidx;
                this.istk[10] = this.istk[this.stkidx];
                return 0;
            }
            case 52: {
                --this.stkidx;
                int v3_13 = this.stkidx - 1;
                Object v0_13 = this.lstk[this.stkidx - 1];
                this.lstk[this.stkidx - 1] = null;
                this.istk[v3_13] = ((byte[]) v0_13)[this.istk[this.stkidx]];
                return 0;
            }
            case 53: {
                --this.stkidx;
                this.istk[10] = this.istk[this.stkidx];
                int v2_38 = this.stkidx;
                this.stkidx = v2_38 + 1;
                this.istk[v2_38] = this.istk[11];
                int v2_39 = this.stkidx;
                this.stkidx = v2_39 + 1;
                this.istk[v2_39] = 3;
                return 0;
            }
            case 54: {
                int v2_40 = this.stkidx;
                this.stkidx = v2_40 + 1;
                this.istk[v2_40] = this.istk[11];
                int v2_41 = this.stkidx;
                this.stkidx = v2_41 + 1;
                this.lstk[v2_41] = this.lstk[6];
                int v3_14 = this.stkidx - 1;
                Object v0_14 = this.lstk[this.stkidx - 1];
                this.lstk[this.stkidx - 1] = null;
                this.istk[v3_14] = ((byte[]) v0_14).length;
                return 0;
            }
            case 55: {
                break;
            }
            default: {
                return opcode;
            }
        }

        this.stkidx += -2;
        if (this.istk[this.stkidx] >= this.istk[this.stkidx + 1]) {
            v0 = 0;
        }

        this.b = v0;
        return 0;
    }
}
