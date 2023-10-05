package sort;

import org.junit.jupiter.api.Order;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.*;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;


/**
 * 다양한 정렬(삽입, 선택, 거품, 퀵, 병합, 힙, 계수, 기수)에 관한 정보를 담은 클래스이다.
 * 구현이 쉬운 정렬은 속도가 느리고, 구현이 어려운 정렬은 속도가 빠르다는 특징이 있다.
 * 평균적으로 n^2의 성능이 나오는 정렬 : 삽입, 선택, 거품
 * n log n : 퀵, 병합, 힙
 * n + a : 계수(n+k, k는 maxValue), 기수(dn, d는 가장 큰 자리수)
 * <p>
 * 안정 정렬 : 삽입, 거품, 병합, 계수, 기수
 * 추가 공간 필요 : 병합, 힙, 계수, 기수
 * <p>
 * 각 정렬 한 줄 설명
 * 삽입 : 정렬된 배열에서 새로운 수 a를 적절한 위치에 삽입하는 정렬
 * 선택 : 정해진 위치 A에 배치한 원소를 선택하는 정렬
 * 거품 : 인접한 두 수를 비교, swap 하며 최종 위치에 적절한 원소를 배치하는 정렬
 * 퀵 : 분할 정복 기법을 적용하여 특정한 원소(pivot)을 적절한 위치(주로 가운데)로 배치하는 정렬
 * 병합 : 분할 정복 기법을 적용하여 정렬된 구간들끼리 비교하며 배치하는 정렬 기법
 * 힙 : 힙 자료구조의 성질을 이용한 정렬 기법
 * 계수 : 각 값들의 개수를 count하여 큰 값부터 배치하는 정렬 기법
 * 기수 : 자리수를 기준으로 오름차순 정렬을 반복하는 정렬 기법
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
public class Sort {

    /*
        N : 10만 N^2 : 10s, N log N : 1.67ms
        N : 100만 : N^2 : 16.7min, n log n : 19ms
     */
    static final int N = 10_0000;
    static final int BOUND = Integer.MAX_VALUE;
    static List<Result> result = new ArrayList<>();
    static Node[] arr;
    static Node[] use;
    static int[] sorted;
    static int size;


    @Setup
    public void beforeAll() {
        shuffledCase();
    }



    public void shuffledCase() {
        Set<Integer> st = new HashSet<>();
        for (int i = 0; i < N; i++) {
            st.add(i);
        }
        List<Integer> shuffler = new ArrayList<>(st);
        Collections.shuffle(shuffler);
        arr = new Node[shuffler.size()];
        sorted = new int[shuffler.size()];
        size = arr.length;
        use = new Node[shuffler.size()];
        for (int i = 0; i < arr.length; i++) {
            Integer value = shuffler.get(i);
            arr[i] = new Node(value, i);
            sorted[i] = value;
        }
        Arrays.sort(sorted);
    }


    @Setup
    public void beforeEach() {
        use = new Node[arr.length];
        for (int i = 0; i < arr.length; i++) {
            use[i] = new Node(arr[i]);
        }
    }

    /**
     * 삽입 정렬 배열 내에서 새로운 선택자(insert)가 기존 배열(0~before)에서 어떤 자리에 위치할 지 정하는 정렬 방식이다. 기존 배열에 얘가 있어야 할 자리에
     * 삽입이 되기 때문에, 삽입 정렬이다. insert(n) *  before(n)이기 때문에 시간 복잡도는 n^2이다. 안정 정렬이다.
     */

    @Benchmark
    public void insertionSort(Blackhole blackhole) {
        blackhole.consume(isSort());
    }

    private Node[] isSort() {
        for (int insert = 1; insert < size; ++insert) {
            for (int before = insert - 1; before >= 0; --before) {
                if (use[before].value > use[before + 1].value) {
                    swap(use, before, before + 1);
                }
            }
        }
        return use;
    }

    /**
     * 선택정렬 정해진 범위(location~end) 내에서 값을 가장 충족시키는 X(최대값 혹은 최소값)를 선택해서 원하는 위치에 배치한다. 특정 자리 A에 있어야 할 X를
     * 선택해서 넣어준다. length * length 만큼 반복하기 때문에 시간복잡도는 O(n^2)
     */
    @Benchmark
    public void selectionSort(Blackhole blackhole) {
        for (int location = 0; location < use.length; ++location) {
            Node min = use[location];
            int minIndex = location;
            for (int select = location + 1; select < use.length; ++select) {
                if (min.value > use[select].value) {
                    min = use[select];
                    minIndex = select;
                }
            }
            swap(use, location, minIndex);
        }
        blackhole.consume(use);
    }

    /**
     * 거품 정렬 인접한 두 원소를 비교하며 정렬하는 알고리즘 배열의 가장 마지막 위치부터 적절한 값이 자리한다. 안정 정렬(stable sort)
     */
    @Benchmark
    public void bubbleSort(Blackhole blackhole) {
        for (int i = 1; i < use.length; ++i) {
            for (int j = 0; j < use.length - i; ++j) {
                if (use[j].value > use[j + 1].value) {
                    swap(use, j, j + 1);
                }
            }
        }
        blackhole.consume(use);

    }

    /*
        분할 정복을 이용해서 pivot을 기준으로 왼쪽은 pivot보다 작은 수들을, 오른쪽은 pivot보다 큰 수들을 배치하며 정렬한다.
        partition을 통해서 pivot을 기준으로 왼쪽에는 pivot보다 작은 수, 오른쪽에는 pivot보다 큰 수를 배치하고,
        pivot의 위치를 확정짓는다.

        시간복잡도 : 일반 케이스는 n log n, worst case : n^2 -> 이미 정렬된 상태일 때,
     */
    @Benchmark
    public void quickSortTest(Blackhole blackhole) {
        quickSort(use, 0, use.length - 1);
        blackhole.consume(use);
    }

    public void quickSort(Node[] arr, int l, int r) {
        if (l < r) {
            int pivot = partition(arr, l, r);
            quickSort(arr, l, pivot - 1);
            quickSort(arr, pivot + 1, r);
        }
    }

    //  pivot이 있을 수 있는 위치를 설정한다.
    int partition(Node[] use, int l, int r) {

        int lo = l;
        int hi = r;
        int pivot = l;        // 부분리스트의 왼쪽 요소를 피벗으로 설정

        // lo가 hi보다 작을 때 까지만 반복한다.
        while (lo < hi) {

            /*
             * hi가 lo보다 크면서, hi의 요소가 pivot보다 작거나 같은 원소를
             * 찾을 떄 까지 hi를 감소시킨다.
             */
            while (use[hi].value > use[pivot].value) {
                hi--;
            }

            /*
             * hi가 lo보다 크면서, lo의 요소가 pivot보다 큰 원소를
             * 찾을 떄 까지 lo를 증가시킨다.
             */
            while (lo < hi && use[lo].value <= use[pivot].value) {
                lo++;
            }
            // 교환 될 두 요소를 찾았으면 두 요소를 바꾼다.
            swap(use, lo, hi);
        }
        /*
         *  마지막으로 맨 처음 pivot으로 설정했던 위치(a[left])의 원소와
         *  lo가 가리키는 원소를 바꾼다.
         */
        swap(use, l, lo);
        // 두 요소가 교환되었다면 피벗이었던 요소는 lo에 위치하므로 lo를 반환한다.
        return lo;
    }


    /**
     * 병합 정렬 분할 정복을 통해 범위를 나누고, 그 범위 내에서 정렬을 진행한다. 범위를 나눌 때에는 가운데를 기준으로 나누고, 정렬된 왼쪽 집단과 정렬된 오른쪽 집단의
     * 가장 작은 수부터 차례로 비교하면서 두 집단을 하나로 합친다.
     * <p>
     * 재귀의 종료 조건은 범위가 1인 경우이다.
     * <p>
     * 시간복잡도는 n log n, worst case도 n log n이다. 안정 정렬이다. 정렬을 위한 추가적인 공간이 필요하다 -> sorted 왼쪽, 오른쪽 배열을
     * 비교하면서 임시 배열에 그 값을 채워줘야함.
     */
    @Benchmark
    @Order(5)
    public void mergeSortTest(Blackhole blackhole) {
        Node[] mergeSorted = new Node[use.length];
        mergeSort(mergeSorted, 0, use.length - 1);
        blackhole.consume(use);
    }

    public void mergeSort(Node[] sorted, int l, int r) {
        if (l >= r) {
            return;
        }

        int mid = (l + r) / 2;
        mergeSort(sorted, l, mid);
        mergeSort(sorted, mid + 1, r);
        merge(sorted, l, mid, r);
        // l

    }

    public void merge(Node[] sorted, int l, int m, int r) {

        int left = l;
        int mid = m + 1;
        int sortedIdx = l;

        while (left <= m && mid <= r) {
            if (use[left].value <= use[mid].value) {
                sorted[sortedIdx++] = use[left++];
            } else {
                sorted[sortedIdx++] = use[mid++];
            }
        }

        while (left <= m) {
            sorted[sortedIdx++] = use[left++];
        }
        while (mid <= r) {
            sorted[sortedIdx++] = use[mid++];
        }

        for (int i = l; i <= r; i++) {
            use[i] = sorted[i];
        }
    }


    /**
     * Heap 자료구조를 이용하여 정렬합니다. 시간복잡도는 n log n이고 정렬을 하기 위한 추가적인 공간이 필요합니다. 불안정 정렬 중 하나입니다. Heap 자료구조의
     * 동작 방식은 다음과 같다. 1. 삽입 과정 1-1 lastIndex에 값을 추가한다. 1-2 현재 위치의 값과 부모의 값과 비교를 하고, 현재 위치의 값이 더 가중치가
     * 높은 값일 경우 둘의 위치를 바꾼다. 1-3 (1-2)과정을 root까지 반복한다. 2. 삭제 과정 2-1 root 값을 제거하고, lastIndex를
     * root(1)위치에 자리하게 한다. 2-2 root부터 자신의 자식들 중 가중치가 높은 값과 가중치를 비교하고, 위치를 바꿔야 한다면 바꾼다. 2-3 (2-2) 과정을
     * leaf 까지 반복한다.
     */
    @Benchmark
    public void heapSortTest(Blackhole blackhole) {
        Node[] sortedNode = new Node[use.length + 1];
        Node[] heap = heapSort(use);
        for (int i = 0; i < use.length; i++) {
            sortedNode[i] = deleteHeap(heap, heap.length - 1 - i);
        }
        blackhole.consume(sortedNode);
    }

    Node[] heapSort(Node[] arr) {
        Node[] heap = new Node[arr.length + 1];
        for (int i = 0; i < arr.length; i++) {
            heapify(heap, i + 1, arr[i]);
        }
        return heap;
    }

    public void heapify(Node[] heap, int index, Node value) {

        heap[index] = value;
        int cur = index;
        while (cur > 1) {
            int parentIndex = cur / 2;
            if (heap[parentIndex].value > heap[cur].value) {
                swap(heap, parentIndex, cur);
                cur = parentIndex;
            } else {
                break;
            }
        }
    }

    public Node deleteHeap(Node[] heap, int lastIndex) {
        Node temp = heap[1];
        heap[1] = heap[lastIndex];
        heap[lastIndex] = null;

        // 재정렬
        int cur = 1;
        while (cur < lastIndex) {

            // index 체크...
            int leftChild = cur * 2;
            int rightChild = cur * 2 + 1;

            if (leftChild > heap.length) {
                break;
            }

            // 둘 다 없는 경우
            if (heap[leftChild] == null && (rightChild >= heap.length
                    || heap[rightChild] == null)) {
                break;
            }
            // 왼쪽만 있는 경우(완전 이진 트리니까)
            else if (heap[leftChild] != null && (rightChild >= heap.length
                    || heap[rightChild] == null)) {
                // left밖에 없으면 여기가 마지막임.
                if (heap[leftChild].value < heap[cur].value) {
                    swap(heap, leftChild, cur);

                }
                break;
            }
            // 둘 다 있는 경우
            else {
                // 둘 중 더 작은 애랑 교환함.
                if (heap[leftChild].value < heap[rightChild].value) {
                    if (heap[leftChild].value <
                            heap[cur].value) {
                        swap(heap, leftChild, cur);
                        cur = leftChild;
                        continue;
                    }

                } else {
                    if (heap[rightChild].value <
                            heap[cur].value) {
                        swap(heap, rightChild, cur);
                        cur = rightChild;
                        continue;
                    }
                }
            }
            break;
        }

        return temp;
    }


    /**
     * 계수 정렬
     * 다른 정렬들이 시간 복잡도가 n이상 발생하는 이유는 비교하는 횟수가 n, log n번씩 발생하기 때문이다.
     * 그래서 비교를 안하고 정렬을 하는 방법으로 시간이 빠르다.
     * 시간복잡도 : O(n + k) k는 maxValue
     * k값이 시간복잡도에 포함되기 때문에, n보다 k가 큰 경우엔 비효율이 발생한다.
     * 정렬을 하기 위한 추가적인 메모리 공간이 필요하다.
     * 안정 정렬이다.
     * 다음과 같은 로직으로 동작한다.
     * 1. 배열을 입력 받을 때 maxValue를 구한다.
     * 2. 0~maxValue 까지의 개수를 기록할 배열을 만든다.
     * 3. 정렬된 값을 넣을 새로운 배열을 만든다.
     * 4. 원본 배열을 탐색하며 기록 배열에 등장 횟수를 기록한다.
     * 5. 기록 배열의 형태를 누적합으로 만든다.
     * 6. 새로운 배열에 다음과 같은 규칙으로 채워 넣는다.
     * 6-1 원본 배열의 뒤부터 하나씩 값을 꺼낸다.
     * 6-2 꺼낸 값이 기록 배열의 누적합을 찾고 새로운배열[누적합-1]에 해당 값을 저장하고, 기록 배열의 누적합을 -1 해준다.
     * 6-3 (1과 2 반복.
     * <p>
     * 6-1, 6-2가 뒤에서부터 작동하기 때문에 안정정렬이 된다.
     */
    @Benchmark
    public void countingSortTest(Blackhole blackhole) {
        int maxValue = -1;
        for (Node node : use) {
            maxValue = Math.max(maxValue, node.value);
        }
        Node[] nodes = countingSort(maxValue);
        blackhole.consume(nodes);

    }

    Node[] countingSort(int maxValue) {
        Node[] arr = new Node[use.length];


        int[] counts = new int[maxValue + 1];
        // 각 수의 개수 기록
        for (int i = 0; i < use.length; i++) {
            int val = use[i].value;
            counts[val]++;
        }


        // count배열의 누적합 생성
        for (int i = 1; i < counts.length; i++) {
            counts[i] += counts[i - 1];
        }

        // 계산하기
        for (int i = use.length - 1; i >= 0; i--) {
            int count = counts[use[i].value]--;
            arr[count - 1] = use[i];
        }
        return arr;
    }


    /**
     * 기수 정렬
     * 계수 정렬과 마찬가지로 비교를 하지 않기 때문에, 성능이 뛰어난 정렬이다.
     * 자리수를 기준으로 정렬을 하기 때문에 가장 큰 자리수를 d라고 했을 때,
     * 시간복잡도는 O(dn)이다.
     * 안정 정렬이고, 정렬을 하기 위한 추가적인 메모리가 필요하다.
     */
    @Benchmark
    public void radixSortTest(Blackhole blackhole) {
        radixSort();
        blackhole.consume(use);
    }

    public void radixSort() {
        // 0~9 까지의 Queue가 필요함.
        Queue<Node>[] queues = new Queue[10];
        for (int i = 0; i < 10; i++) {
            queues[i] = new ArrayDeque<>();
        }

        // 자리수로 정렬을 한다.

        int n = 1;
        while (true) {

            for (int i = 0; i < use.length; i++) {
                queues[(use[i].value / n) % 10].add(use[i]);
            }

            if (queues[0].size() == use.length) break;
            // 재배치
            int index = 0;
            for (Queue<Node> queue : queues) {
                while (!queue.isEmpty())
                    use[index++] = queue.poll();
            }
            n *= 10;
        }
    }


    static public void swap(Node[] arr, int l, int r) {
        Node temp = arr[r];
        arr[r] = arr[l];
        arr[l] = temp;
    }


    public static class Node {

        int value;
        int index;

        public Node(int value, int index) {
            this.value = value;
            this.index = index;
        }

        public Node(Node origin) {
            this.value = origin.value;
            this.index = origin.index;
        }

        @Override
        public String toString() {
            return "Node{" +
                    "value=" + value +
                    ", index=" + index +
                    '}';
        }
    }

    public static class Result {
        String name;
        long time;

        public Result(String name, long time) {
            this.name = name;
            this.time = time;
        }

        @Override
        public String toString() {
            return name + " : " + time + " (ns)";
        }
    }
}
