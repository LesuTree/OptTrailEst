package Main;

import javax.swing.text.Utilities;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class OptTrailEst {

    // 定义Present算法的S盒
    private static byte[] sBox = new byte[] {
            0x05,0x0e,0x0f,0x08,
            0x0c,0x01,0x02,0x0d,
            0x0b,0x04,0x06,0x03,
            0x00,0x07,0x09,0x0a
    };
     private static int[][] differentialTable = generateDifferentialTable();
    private static byte[][] beta;

    private static byte[][] alpha;
    private static byte maxByte;
    // 定义Present算法的轮数
    private static final int ROUNDS = 4;

    // 存储最终特征的列表
    private static List<Feature> optimalCharacteristic = new ArrayList<>();

    // 存储每一轮的pRd值
    private static double[] pRd;
    private static double[][] pRdMatrix;

    // 存储每一轮的α值
    private static int[] alphaValues;

    // 存储每一轮的β值
    private static int[] betaValues;

    // 存储每一轮的QRi值
    private static int[] QRiValues;

    // 存储每一轮的PEstim值
    private static double pEstim;

    public static void main(String[] args) {
        // 初始化数据结构
        pRd = new double[ROUNDS];
        alphaValues = new int[ROUNDS + 1];
        betaValues = new int[ROUNDS + 1];
        QRiValues = new int[ROUNDS + 1];

        // 调用OptTrailEst算法
        optTrailEstimation(ROUNDS,32);

        // 打印结果
        System.out.println("Optimal Characteristic:");
        for (Feature feature : optimalCharacteristic) {
            System.out.println("Round " + feature.round + ": (α" + feature.alpha + ", β" + feature.beta + ")");
        }
    }

    /**
     * Algorithm OptTrailEst For n from 1 to N,
     * If p[n]-SB is lower than the rank-one bound, then Exit the loop
     * Else,
     * For each output difference β1 activating n S-boxes,
     * Call FirstRound()
     * If a characteristic has been found (E is not empty), then
     * Return E and pEstim Else
     * Return ()
     * R rounds of present,N number of s box in each round
     */
    private static void optTrailEstimation(int R,int N){
        for(int n=1;n<N+1;n++){
            if (pSB(n)< rankBound(1)){
                break;
            }
            else {
                // For each output difference β1 activating n S-boxes,
                int[] selectedIndex = new int[n];
                activateNSBoxes(sBox,n,0xf,0,selectedIndex); // 递归调用， 包含了FirstRound

            }
        }
    }
    private static void activateNSBoxes(byte[] sBoxes, int n, int F, int start, int[] selectedIndex) {
        if (n==0){
            processCombination(sBoxes,selectedIndex);
            return;
        }
        for(int i = start; i<= sBoxes.length-n; i++){
            selectedIndex[selectedIndex.length - n] = i;
            activateNSBoxes(sBoxes,n-1,F,i+1, selectedIndex);
        }
    }

    private static void processCombination(byte[] sBoxes, int[] selectedIndex) {
        // 在这里进行生成的组合的处理，可以打印、保存等操作
        System.out.println(Arrays.toString(selectedIndex));
        byte current = 0;
        boolean flag = true;
        int n = 2 << (selectedIndex.length * 4 - 1);
        for (int i = 1; i<=n;i++){
            for (int j = 0; j<selectedIndex.length;j++){
                flag = true;
                current = (byte)((i >> (j * 4)) & 0xF);
                if (current == 0){
                    flag = false;
                    break;
                }
                beta[0][selectedIndex[j]] = current;
            }
            if (flag) {
                System.out.println(Arrays.toString(sBoxes));
                //TODO 调用函数
                searchProcedureFirstRound(beta[0]);
            }
        }
    }
    private static double pSB(int n) {
        return 0;
    }


    /**
     * pRd(1) ← maxα P(α → β1)
     * α1 ←αsuchthatP(α→β1)=pRd(1)
     * α2 ← π(β1)
     * If R > 2, then
     * Call the search procedure Round(2), Else
     * Call the search procedure LastRound() End of the function
     * @param beta0
     */
    private static void searchProcedureFirstRound(byte[] beta0) {
        pRd[0] = calculateMaxAlpha(beta0);
        alpha[0] = calculateMaxAlpha2(beta0);
        alpha[1] = presentEncrypt.pExchange(beta0);
        if (ROUNDS > 2) {
            searchProcedureRound(2);
        } else {
            searchProcedureLastRound();
        }
    }

    /**
     * Function Round(r) (2 ≤ r < R)
     * βr ←0SN
     * For each i such that 1 ≤ i ≤ #SB(αr)
     * Let ρr(i) denote the position of the i-th S-box activated by αr
     * Call SubRound(r,1)
     * End of the function
     * @param r
     */

    private static void searchProcedureRound(int r) {
        // beta[r-1] = new byte[16];
        int _SB = getActivateNumber(alpha[r-1]);
        int[] _ActivateIndex = getActivateIndex(alpha[r-1],_SB);
        //For each i such that 1 ≤ i ≤ #SB(αr)
        for (int i = 1;i<=_SB;i++){
            //Call SubRound(r,1)
            searchProcedureSubRound(r,1,_SB,_ActivateIndex);
        }
    }

    /**
     * Function SubRound(r,n)
     * If n > #SB(αr), then
     * pRd(r) ← P(αr → βr) = Q#SB(αr) pRd(r,j); j=1
     * αr+1 ← π(βr+1) If r + 1 < R, then
     * Call Round(r + 1) Else
     * Call LastRound() Else
     * ◃ We continue Round(r − 1) or FirstRound()
     * For each brρr(n) sorted in decreasing order according to Pρr(n)(arρr(n) → ·)
     * Lr,n ←list(αr)∧(0ρr(n)1N−ρr(n))
     * pRd(r,n) ← Pρr(n) arρr(n) → brρr(n)
     * If Qr−1 pRd(i) · Qn pRd(r,j) · pListSB(Lr,n) is lower than the rank-r bound, then i=1 j=1
     * Exit the loop
     * If π is a bit permutation, then
     * L′r,n ← Wni=1 list(π(Dρr(i)(brρr(i))));
     * If Qr−1 pRd(i) ·Qn pRd(r,j) ·pListSB(Lr,n) ·pListSB(L′
     * ◃ See Theorem 18
     * ) is not lower than the rank-(r+1) ◃ See Theorem 25
     * Else
     * End of the function
     * ◃ We continue SubRound(r,n − 1) or Round(r) Fig. 4. Second optimization – the search function Round
     * i=1 j=1 bound, then
     * r,n
     * Call SubRound(r,n + 1) Call SubRou
     * @param r
     * @param n
     */
    private static void searchProcedureSubRound(int r, int n, int _SB,int[] ActivateIndex){
        if (n>_SB){
            pRd[r-1] = 1;
            for(int i = 0;i<_SB;i++){
                pRd[r-1] *= pRdMatrix[r-1][i];
            }
            alpha[r] = presentEncrypt.pExchange(beta[r-1]);
            if (r + 1 < ROUNDS){
                searchProcedureRound(r+1);
            }
            else
                searchProcedureLastRound();
        }
        else{
            //For each brρr(n) sorted in decreasing order according to Pρr(n)(arρr(n) → ·)
            int[] decreasingIndex = getDecreasingOrderIndex (alpha[r-1][ActivateIndex[n-1]]);
            int[] L = getL(alpha[r-1],ActivateIndex[n-1]);
            for (int i = 0;i<decreasingIndex.length;i++){

                pRdMatrix[r-1][n-1] = differentialTable[alpha[r-1][ActivateIndex[n-1]]][decreasingIndex[i]];
                double pro_0_r_2 = Uitils.multiply(pRd,0,r-2);
                double pro_0_n_1 = Uitils.multiply(pRdMatrix[r-1],0,n-1);
                double pro_ListSB = getPListSB(L);
                if (pro_ListSB*pro_0_n_1*pro_0_r_2 < rankBound(r)){
                    break;
                }
                int[] L_r = getL_r(n,ActivateIndex, beta[r-1],r);
                if (pro_ListSB*pro_ListSB*pro_0_n_1*getPListSB(L)*getPListSB(L_r) >= rankBound(r+1)){
                    beta[r-1][ActivateIndex[n-1]] = (byte) decreasingIndex[i];
                    searchProcedureSubRound(r, n+1, _SB,ActivateIndex);
                }
            }
        }
    }

    private static int[] getL_r(int n, int[] activateIndex, byte[] beta_r_1,int r) {
        alpha[r] = presentEncrypt.pExchange(beta_r_1);
        return getList(alpha[r]);
    }

    private static double getPListSB(int[] L){
        double probability = 1;

        for(int i = 0; i<L.length;i++){
            if(L[i] != 0){
                probability *= (double) findMaxValueExcludingFirstRow(differentialTable)/(double) 16;
            }
        }
        return probability;
    }

    private static int findMaxValueExcludingFirstRow(int[][] array) {
        int max = Integer.MIN_VALUE;

        for (int i = 1; i < array.length; i++) {
            for (int value : array[i]) {
                if (value > max) {
                    max = value;
                }
            }
        }

        return max;
    }

    private static int[] getL(byte[] alpha, int n){
        int[] list = getList(alpha);
        for (int i = 0; i<n;i++){
            list[i] = 0;
        }
        return list;
    }
    private static int[] getList(byte[] alpha){
        int[] list = new int[alpha.length];
        for (int i = 0;i<alpha.length;i++){
            list[i] = alpha[1]==0?0:1;
         }
        return list;
    }
    private static int[] sortAndGetIndices(int[] array) {
        // 创建一个包含原始下标和对应值的数组
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }

        // 使用自定义比较器，根据元素值进行降序排序
        Arrays.sort(indices, Comparator.comparingInt(index -> array[(int) index]).reversed());

        // 将 Integer 数组转换为 int 数组
        return Arrays.stream(indices).mapToInt(Integer::intValue).toArray();
    }
    private static int[] getDecreasingOrderIndex(byte b) {
        int[] outDiff = differentialTable[b];
        int[] sortedDiffIndices = sortAndGetIndices(outDiff);
        return sortedDiffIndices;
    }

    private static int getActivateNumber(byte[] alpha){
        int count = 0;
        for (int i = 0; i<alpha.length;i++) {
            if (alpha[i] != 0){
                count++;
            }
        }
        return count;
    }
    private static int[] getActivateIndex(byte[] alpha, int number){
        int[] activateIndex = new int[number];
        int count = 0;
        for (int i = 0;i<alpha.length;i++){
            if (alpha[i]!=0){
                activateIndex[count++] = alpha[i];
            }
        }
        return activateIndex;
    }
    /**
     *Function LastRound()
     * pRd(R) ← maxβ P(αR → β)
     * βR ←βsuchP(αR →β)=pRd(R)
     * If QRi=1 pRd(i) ≥ pEstim, then
     * E ← ((α1, β1), . . . , (αR, βR))
     * pEstim ← QRi=1 pRd(i) = P(E). End of the function
     */
    private static void searchProcedureLastRound() {
        pRd[ROUNDS-1] = calculateMaxBeta(alpha[ROUNDS-1]);
        beta[ROUNDS-1] = calculateMaxBeta2(alpha[ROUNDS-1]);
        if (Uitils.multiply(pRd,0,ROUNDS-1)>=pEstim){
            saveCharacteristic();
            pEstim = Uitils.multiply(pRd,0,ROUNDS-1);
        }

        if (QRiValues[ROUNDS] == 1 && pRd[ROUNDS] >= pEstim) {
            saveCharacteristic();
        }
    }


    //根据beta数组，寻找最大的概率
    private static double calculateMaxAlpha(byte[] beta) {
        double maxPRd = 1;
        byte[] maxAlpha = new byte[beta.length];
        for (int i = 0; i < beta.length; i++) {
            int[] currentPRdAlpha = calculateMaxPRdAlpha(beta[i]);
            int currentPRd = currentPRdAlpha[0];
            int currentAlpha = currentPRdAlpha[1];
            if (currentPRd != 0) {
                maxPRd *= ((double) currentPRd/(double) 16);
                maxAlpha[i] = (byte) currentAlpha;
            }
        }
        return maxPRd;
    }

    private static byte[] calculateMaxAlpha2(byte[] beta) {
        double maxPRd = 1;
        byte[] maxAlpha = new byte[beta.length];
        for (int i = 0; i < beta.length; i++) {
            int[] currentPRdAlpha = calculateMaxPRdAlpha(beta[i]);
            int currentPRd = currentPRdAlpha[0];
            int currentAlpha = currentPRdAlpha[1];
            if (currentPRd != 0) {
                maxPRd *= ((double) currentPRd/(double) 16);
                maxAlpha[i] = (byte) currentAlpha;
            }
        }
        return maxAlpha;

    }
    private static int[] calculateMaxPRdAlpha(int beta) {
        int maxPRd = 0;
        int maxAlpha = 0;
        for (int alpha = 1; alpha < 16; alpha++) {
            int currentPRd = differentialTable[alpha][beta];
            if (maxPRd < currentPRd){
                maxPRd = currentPRd;
                maxAlpha = alpha;
            }
        }
        int result[] = new int[]{maxPRd,maxAlpha};
        return result;
    }

    private static double calculateMaxBeta(byte[] alpha) {
        double maxPRd = 1;
        byte[] maxBeta = new byte[alpha.length];
        for (int i = 0; i < alpha.length; i++) {
            int[] currentPRdBeta = calculateMaxPRdBeta(alpha[i]);
            int currentPRd = currentPRdBeta[0];
            int currentBeta = currentPRdBeta[1];
            if (currentPRd != 0) {
                maxPRd *= ((double) currentPRd/(double) 16);
                maxBeta[i] = (byte) currentBeta;
            }
        }
        return maxPRd;

    }
    private static byte[] calculateMaxBeta2(byte[] alpha) {
        double maxPRd = 1;
        byte[] maxBeta = new byte[alpha.length];
        for (int i = 0; i < alpha.length; i++) {
            int[] currentPRdBeta = calculateMaxPRdBeta(alpha[i]);
            int currentPRd = currentPRdBeta[0];
            int currentBeta = currentPRdBeta[1];
            if (currentPRd != 0) {
                maxPRd *= ((double) currentPRd/(double) 16);
                maxBeta[i] = (byte) currentBeta;
            }
        }
        return maxBeta;

    }
    private static int[] calculateMaxPRdBeta(int alpha) {
        int maxPRd = 0;
        int maxBeta = 0;
        for (int beta = 1; beta < 16; beta++) {
            int currentPRd = differentialTable[alpha][beta];
            if (maxPRd < currentPRd){
                maxPRd = currentPRd;
                maxBeta = beta;
            }
        }
        int result[] = new int[]{maxPRd,maxBeta};
        return result;
    }
    // 生成S盒差分分布表
    private static int[][] generateDifferentialTable() {
        int[][] differentialTable = new int[16][16];

        for (int inputDiff = 0; inputDiff < 16; inputDiff++) {
            int count = 0;
            for (int input = 0; input < 16; input++) {
                int output = sBox[input];
                int inputDiffActual = input ^ inputDiff;
                int outputDiff =sBox[inputDiffActual];
                int outputDiffActual = output ^ outputDiff;
                differentialTable[inputDiff][outputDiffActual]++;

            }
        }

        return differentialTable;
    }

    private static int calculatePRd(int r, int beta, int alpha) {
        int inDiff = alpha;
        int outDiff = beta;
        int inMasked = inDiff ^ sBox[0];
        int outMasked = outDiff ^ sBox[0];
        int inDiffMasked = sBox[inMasked] ^ sBox[inMasked ^ inDiff];
        int outDiffMasked = sBox[outMasked] ^ sBox[outMasked ^ outDiff];
        return inDiffMasked ^ outDiffMasked;
    }

    private static int findAlpha(int beta, int maxPRd) {
        for (int alpha = 0; alpha < 16; alpha++) {
            if (calculatePRd(1, beta, alpha) == maxPRd) {
                return alpha;
            }
        }
        return 0;
    }

    private static int sBoxPermutation(int input) {
        return sBox[input];
    }


    private static double rankBound(int r) {
        return pEstim / productPRd(ROUNDS - r);
    }

    private static int productPRd(int r) {
        int product = 1;
        for (int i = 1; i <= r; i++) {
            product *= pRd[i];
        }
        return product;
    }

    private static void saveCharacteristic() {
        List<Feature> characteristic = new ArrayList<>();
        for (int i = 1; i <= ROUNDS; i++) {
            characteristic.add(new Feature(i, alphaValues[i], betaValues[i]));
        }

        optimalCharacteristic = characteristic;
        pEstim = productPRd(ROUNDS);
    }

    // 用于存储特征的简单类
    static class Feature {
        int round;
        int alpha;
        int beta;

        Feature(int round, int alpha, int beta) {
            this.round = round;
            this.alpha = alpha;
            this.beta = beta;
        }
    }
}
