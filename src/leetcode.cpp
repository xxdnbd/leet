#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <set>
#include <sstream>
#include <queue>
#include <climits>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
using namespace std;

struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}
};

struct TreeNode {
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:

	// Two Sum
    vector<int> twoSum(vector<int> &numbers, int target) {
    	int i = 0, j = numbers.size() - 1;
    	vector<int> ret;
    	vector<int> sorted(numbers);
    	sort(sorted.begin(), sorted.end());
    	while (i < j) {
    		int cur_sum = sorted.at(i) + sorted.at(j);
    		if (cur_sum < target)
    			i++;
    		else if (cur_sum > target)
    			j--;
    		else
    			break;
    	}
    	size_t index_i, index_j;
    	for (size_t k = 0; k < numbers.size(); ++k) {
    		if (sorted.at(i) == numbers.at(k)) {
    			index_i = k + 1;
    			break;
    		}
    	}
    	for (size_t k = 0; k < numbers.size(); ++k) {
    		if (sorted.at(j) == numbers.at(k) && (index_i != k + 1)) {
    			index_j = k + 1;
    			break;
    		}
    	}
    	if (index_i > index_j)
    		swap(index_i, index_j);
    	ret.push_back(index_i);
    	ret.push_back(index_j);
    	return ret;
    }

    // Longest Substring Without Repeating Characters
    int lengthOfLongestSubstring(string s) {
    	int maxLength = 0, left_pos = 0;
    	int char_count[255] = {0};
    	if (s.length() <= 1)
    		return s.length();
    	char_count[s[left_pos]]++;
    	maxLength = 1;
    	for (int right_pos = 1; right_pos < s.length(); ++right_pos) {
    		while (left_pos < right_pos && char_count[s[right_pos]] != 0)
    			char_count[s[left_pos++]]--;
    		char_count[s[right_pos]]++;
    		if (right_pos - left_pos + 1 > maxLength)
    			maxLength = right_pos - left_pos + 1;
    	}
    	return maxLength;
    }

    // Add Two Numbers
    ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
    	ListNode *ret_head = new ListNode(0);
    	ListNode *ret_current = ret_head;
    	int carry = 0;
    	ListNode *l1_cur = l1, *l2_cur = l2;
    	while (l1_cur != NULL || l2_cur != NULL || carry != 0) {
    		int sum = carry;
    		if (l1_cur != NULL) {
    			sum += l1_cur->val;
    			l1_cur = l1_cur->next;
    		}
    		if (l2_cur != NULL) {
    			sum += l2_cur->val;
    			l2_cur = l2_cur->next;
    		}
    		carry = sum / 10;
    		ListNode *t = new ListNode(sum % 10);
    		ret_current->next = t;
    		ret_current = t;
    	}
    	ListNode *ret = ret_head->next;
    	delete ret_head;
    	return ret;
    }

    // Reverse Integer
    int reverse(int x) {
    	bool sign = x > 0 ? true : false;
    	if (x < 0)
    		x = -x;
    	int len_factor = 1;
    	while (x / len_factor >= 10)
    		len_factor *= 10;
    	int low_factor = 1;
    	int high_factor = len_factor;
    	int ret = 0;
    	while (low_factor < high_factor) {
    		int low_digit = x / low_factor % 10;
    		int high_digit = x / high_factor % 10;
    		ret += low_digit * high_factor + high_digit * low_factor;
    		low_factor *= 10;
    		high_factor /= 10;
    	}
    	if (low_factor == high_factor)
    		ret += x / low_factor % 10 * low_factor;
    	return sign ? ret : -ret;
    }

    // String to Integer (atoi)
    int atoi(const char *str) {
    	const char *p = str;
    	long long ret = 0;
    	bool sign = true;
    	while (*p != '\0' && *p == ' ')
    		++p;
    	if (*p == '-') {
    		sign = false;
    		++p;
    	}
    	else if (*p == '+')
    		++p;

    	while (*p != '\0') {
    		if (*p == ' ' || *p < '0' || *p > '9')
    			break;
    		ret = ret * 10 + (*p - '0');
    		++p;
    	}
    	ret = sign ? ret : -ret;
    	if (ret > 2147483647)
    		return 2147483647;
    	else if (ret < -2147483648L)
    		return -2147483648L;
    	else
    		return (int)ret;
    }

    // Palindrome Number
    bool isPalindrome(int x) {
    	if (x < 0)
    		return false;
    	int len_factor = 1;
    	while (x / len_factor >= 10)
    		len_factor *= 10;
    	int low_factor = 1;
    	int high_factor = len_factor;
    	int ret = 0;
    	while (low_factor < high_factor) {
    		int low_digit = x / low_factor % 10;
    		int high_digit = x / high_factor % 10;
    		if (low_digit != high_digit)
    			return false;
    		low_factor *= 10;
    		high_factor /= 10;
    	}
    	return true;
    }

    // Regular Expression Matching
    bool isMatch(const char *s, const char *p) {
    	if (*s == '\0' && *p == '\0')
    		return true;
    	else if (*s !='\0' && *p == '\0')
    		return false;

    	char t = *p;
    	char next = *(p + 1);
    	if (next != '*') {
    		if (*s !='\0' && (t == '.' || t == *s))
    			return isMatch(s + 1, p + 1);
    		else
    			return false;
    	}
    	else {
    		if (isMatch(s, p + 2))
    			return true;
    		while (*s != '\0' && (*s == t || t == '.')) {
    			if (isMatch(s + 1, p + 2))
    				return true;
    			s++;
    		}
    		return false;
    	}
    }

    // Largest Rectangle in Histogram
    int largestRectangleArea(vector<int> &height) {
    	int *left_max = new int[height.size()];
    	int *right_max = new int[height.size()];
    	for (int i = 0; i < height.size(); ++i) {
    		left_max[i] = i;
    		int j = i - 1;
    		while (j >= 0 && height[j] >= height[i]) {
    			left_max[i] = left_max[j];
    			j = left_max[j];
    			--j;
    		}
    	}
    	for (int i = height.size() - 1; i >= 0; --i) {
    		right_max[i] = i;
    		int j = i + 1;
    		while (j < height.size() && height[j] >= height[i]) {
    			right_max[i] = right_max[j];
    			j = right_max[j];
    			++j;
    		}
    	}

    	int max_area = 0;
    	for (int i = 0; i < height.size(); ++i) {
    		int area = height[i] * (right_max[i] - left_max[i] + 1);
    		if (area > max_area)
    			max_area = area;
    	}
    	delete [] left_max;
    	delete [] right_max;
    	return max_area;
    }

    // Container With Most Water
    int maxArea(vector<int> &height) {
    	int i = 0, j = height.size() - 1;
    	int max_area = 0;
    	while (i < j) {
    		int cur_area = min(height[i], height[j]) * (j - i);
    		max_area = max(cur_area, max_area);
    		if (height[i] < height[j])
    			i++;
    		else
    			j--;
    	}
    	return max_area;
    }

    // Longest Common Prefix
    string longestCommonPrefix(vector<string> &strs) {
    	if (strs.size() == 0)
    		return "";
    	size_t min_length = strs[0].length();
    	string ret;
    	for (int i = 0; i < strs.size(); ++i)
    		min_length = min(strs[i].length(), min_length);

    	for (int i = 0; i < min_length; ++i) {
    		bool is_same = true;
    		for (int j = 0; j < strs.size(); ++j) {
    			if (strs[0][i] != strs[j][i]) {
    				is_same = false;
    				break;
    			}
    		}
    		if (!is_same)
    			break;
    		ret.push_back(strs[0][i]);
    	}
    	return ret;
    }

    // 3Sum
    vector<vector<int> > threeSum(vector<int> &num) {
    	vector<vector<int> > *ret = new vector<vector<int> >();
    	set<string> *dup = new set<string>();
    	if (num.size() < 3)
    		return *ret;
    	sort(num.begin(), num.end());
    	for (int k = 1; k < num.size() - 1; ++k) {
    		int mid = num.at(k);
    		int i = 0, j = num.size() - 1;
    		while (i < k && k < j) {
    			int res = num[i] + num[k] + num[j];
    			if (res > 0)
    				--j;
    			else if (res < 0)
    				++i;
    			else {
    				ostringstream ostream;
    				ostream << num[i] << " " << num[k] << " " << num[j];
    				string pat = ostream.str();
    				if (dup->find(pat) == dup->end()) {
    					dup->insert(pat);
    					vector<int> three;
    					three.push_back(num[i]);
    					three.push_back(num[k]);
    					three.push_back(num[j]);
    					ret->push_back(three);
    				}
    				--j;
    				++i;
    			}
    		}
    	}
    	delete dup;
    	return *ret;
    }

    // Letter Combinations of a Phone Number
    void letterCombinationHelper(string digits, vector<string> *results, string current,
    		                     string phoneMaps[]) {
    	if (digits.empty()) {
    		results->push_back(current);
    		return;
    	}

    	char x = digits[0] - '0';
    	string left = digits.substr(1);
    	for (int i = 0; i < phoneMaps[x].length(); i++)
    		letterCombinationHelper(left, results, current + phoneMaps[x][i], phoneMaps);

    }
    vector<string> letterCombinations(string digits) {
    	string phoneMaps[10] = {"", "", "abc", "def", "ghi", "jkl",
    							"mno", "pqrs", "tuv", "wxyz"};
    	vector<string>* results = new vector<string>();
    	letterCombinationHelper(digits, results, "", phoneMaps);
    	return *results;
    }

    // Remove Nth Node From End of List
    ListNode *removeNthFromEnd(ListNode *head, int n) {
    	if (head == NULL)
    		return head;
    	ListNode *last = head, *first = head, *firstPre = NULL;
    	for (int i = 0; i < n - 1; ++i)
    		last = last->next;
    	while (last->next) {
    		firstPre = first;
    		first = first->next;
    		last = last->next;
    	}
    	if (firstPre) {
    		firstPre->next = first->next;
    		delete first;
    	}
    	else {
    		head = first->next;
    		delete first;
    	}
    	return head;
    }

    // Valid Parentheses
    bool isValid(string s) {
    	vector<char> stack;
    	for (int i = 0; i < s.length(); ++i) {
    		char c = s[i];
    		if (c == '(' || c == '{' || c == '[') {
    			stack.push_back(c);
    			continue;
    		}
    		if (stack.size() == 0)
    			return false;
    		char c_in_stack = stack[stack.size() - 1];
    		stack.pop_back();
    		if ((c - c_in_stack) == 1 || (c - c_in_stack) == 2)
    			continue;

    		return false;
    	}
    	if (stack.size() == 0)
    		return true;
    	return false;
    }


    // Generate Parentheses
    void generateParenthesisHelper(int cur_left, int cur_right, int n,
    		                       string current, vector<string>* ret) {
    	if (cur_left == n && cur_right == n) {
    		ret->push_back(current);
    		return;
    	}
    	if (cur_left < n)
    		generateParenthesisHelper(cur_left + 1, cur_right, n, current + "(", ret);
    	if (cur_right < cur_left)
    		generateParenthesisHelper(cur_left, cur_right + 1, n, current + ")", ret);
    }
    vector<string> generateParenthesis(int n) {
    	vector<string> ret;
    	generateParenthesisHelper(0, 0, n, "", &ret);
    	return ret;
    }

    // Merge k Sorted Lists
    struct compareList {
        bool operator()(const ListNode& x, const ListNode& y) {
        	return x.val > y.val;
        };
    };

    ListNode *mergeKLists(vector<ListNode *> &lists) {
    	priority_queue<ListNode, vector<ListNode>, compareList> pqueue;
    	for (int i = 0; i < lists.size(); ++i) {
    		if (lists[i])
    			pqueue.push(*lists[i]);
    	}
    	ListNode *head = new ListNode(0);
    	ListNode *tail = head;
    	while (!pqueue.empty()) {
    		ListNode x = pqueue.top();
    		pqueue.pop();
    		if (x.next != NULL)
    			pqueue.push(*x.next);
    		ListNode* copyx = new ListNode(x.val);
    		tail->next = copyx;
    		tail = tail->next;
    	}
    	return head->next;
    }

    // Swap Nodes in Pairs
    ListNode *swapPairs(ListNode *head) {
		if (head == NULL || head->next == NULL)
			return head;
		ListNode *preFirst = new ListNode(0);
		preFirst->next = head;
		ListNode *first = head;
		ListNode *second = head->next;
		ListNode *virtualHead = preFirst;
		while (preFirst->next != NULL && preFirst->next->next != NULL) {
			preFirst->next = second;
			first->next = second->next;
			second->next = first;
			preFirst = first;
			if (preFirst->next != NULL) {
				first = preFirst->next;
				second = first->next;
			}
		}
		return virtualHead->next;
    }

	// Validate Binary Search Tree
	struct valMinMax {
		int minVal;
		int maxVal;
	};
	
	bool isValidBSTHelper(TreeNode *root, valMinMax& v) {
		if (root == NULL)
			return true;
		bool isValid = true;
		valMinMax leftVals, rightVals;
		v.minVal = v.maxVal = root->val;
		if (root->left) {
			isValid &= isValidBSTHelper(root->left, leftVals);
			isValid &= root->val > leftVals.maxVal;
			v.minVal = min(leftVals.minVal, v.minVal);
		}
		if (root->right) {
			isValid &= isValidBSTHelper(root->right, rightVals);
			isValid &= root->val < rightVals.minVal;
			v.maxVal = max(rightVals.maxVal, v.maxVal);
		}
		return isValid;
	}
	
	bool isValidBST(TreeNode *root) {
		valMinMax vals;
		return isValidBSTHelper(root, vals);
	}

	// Sqrt(x)
	int sqrt(int x) {
		if (x <= 1)
			return x;
		int left = 1, right = x;
		while (left + 1 < right) {
			int mid = left + (right - left) / 2;
			if (x / mid < mid)
				right = mid;
			else if (x / mid > mid)
				left = mid;
			else
				return mid;
		}
		if (x / right == right)
			return right;
		else
			return left;
	}

	// Balanced Binary Tree
	int tmp_depth;
	bool isBalanced(TreeNode *root) {
		if (root == NULL) {
			tmp_depth = -1;
			return true;
		}
		bool isbalance = true;
		isbalance &= isBalanced(root->left);
		int left_depth = tmp_depth;
		isbalance &= isBalanced(root->right);
		int right_depth = tmp_depth;

		isbalance &= abs(right_depth - left_depth) <= 1;
		tmp_depth = max(left_depth, right_depth) + 1;
		return isbalance;
	}

	// Pascal's Triangle II
	vector<int> getRow(int rowIndex) {
		vector<int> lastRow;
		vector<int> curRow;
		curRow.push_back(1);
		for (int i = 1; i <= rowIndex; ++i) {
			lastRow = curRow;
			curRow.clear();
			curRow.push_back(1);
			for (int j = 0; j < lastRow.size() - 1; ++j)
				curRow.push_back(lastRow[j] + lastRow[j + 1]);
			curRow.push_back(1);
		}
		return curRow;
	}

	// Pascal's Triangle
    vector<vector<int> > generate(int numRows) {
        vector<vector<int> > ret;
		vector<int> lastRow;
		vector<int> curRow;
        if (numRows == 0)
            return ret;
		curRow.push_back(1);
        ret.push_back(curRow);
		for (int i = 2; i <= numRows; ++i) {
			lastRow = curRow;
			curRow.clear();
			curRow.push_back(1);
			for (int j = 0; j < lastRow.size() - 1; ++j)
				curRow.push_back(lastRow[j] + lastRow[j + 1]);
			curRow.push_back(1);
            ret.push_back(curRow);
		}
		return ret;
    }


	// Minimum Depth of Binary Tree
	void minDepthHelper(TreeNode *root, int curDepth, int* minDepth) {
		if (root == NULL)
			return;
		else if (root->left == NULL && root->right == NULL) {
			*minDepth = min(curDepth + 1, *minDepth);
			return;
		}
		else {
			minDepthHelper(root->left, curDepth + 1, minDepth);
			minDepthHelper(root->right, curDepth + 1, minDepth);
		}
	}
	int minDepth(TreeNode *root) {
		int min_dep = INT_MAX;
		if (root == NULL)
			return 0;
		minDepthHelper(root, 0, &min_dep);
		return min_dep;
	}

	// Rotate List
	ListNode *rotateRight(ListNode *head, int k) {
		if (head == NULL)
			return NULL;
		ListNode *fast = head, *slow = head;
		int length = 0;
		while (fast) {
			fast = fast->next;
			++length;
		}
		k = k % length;
		if (k == 0 || length == 1)
			return head;
		fast = head;
		for (int i = 0; i < k; ++i)
			fast = fast->next;
		while (fast->next) {
			fast = fast->next;
			slow = slow->next;
		}
		fast->next = head;
		ListNode *ret = slow->next;
		slow->next = NULL;
		return ret;
	}

	// Search for a Range
	vector<int> searchRange(int A[], int n, int target) {
		// Start typing your C/C++ solution below
		// DO NOT write int main() function
		vector<int> ret;
		if (n == 0) {
			ret.push_back(-1);
			ret.push_back(-1);
			return ret;
		}
		int i = 0, j = n - 1;
		int left, right;
		while (i + 1 < j) {
			int mid = i + (j - i) / 2;
			if (A[mid] >= target)
				j = mid;
			else
				i = mid;
		}
		if (A[i] == target)
			left = i;
		else
			left = j;
		i = 0;
		j = n - 1;
		while (i + 1 < j) {
			int mid = i + (j - i) / 2;
			if (A[mid] <= target)
				i = mid;
			else
				j = mid;
		}
		if (A[j] == target)
			right = j;
		else
			right = i;

		if (A[left] != target || A[right] != target) {
			left = -1;
			right = -1;
		}
		ret.push_back(left);
		ret.push_back(right);
		return ret;
	}

	// Symmetric Tree
	bool isSymmetricHelper(TreeNode *t1, TreeNode *t2) {
		if (t1 == NULL && t2 == NULL)
			return true;
		if ((t1 == NULL && t2 != NULL) || (t1 != NULL && t2 == NULL))
			return false;
		if (t1->val != t2->val)
			return false;
		return isSymmetricHelper(t1->left, t2->right) && 
			isSymmetricHelper(t1->right, t2->left);
	}

	bool isSymmetric(TreeNode *root) {
		if (root == NULL)
			return true;
		else
			return isSymmetricHelper(root->left, root->right);
	}

	// Binary Tree Level Order Traversal
    vector<vector<int> > levelOrder(TreeNode *root) {
       vector<vector<int> > ret;
       if (root == NULL)
           return ret;
       vector<TreeNode *> next_level_node;
       vector<TreeNode *> cur_level_node;
       next_level_node.push_back(root);
       while (!next_level_node.empty()) {
           vector<int> cur_level_value;
           cur_level_node = next_level_node;
           next_level_node.clear();
           for (int i = 0; i < cur_level_node.size(); ++i) {
               cur_level_value.push_back(cur_level_node[i]->val);
               if (cur_level_node[i]->left != NULL)
                    next_level_node.push_back(cur_level_node[i]->left);
               if (cur_level_node[i]->right != NULL)
                    next_level_node.push_back(cur_level_node[i]->right);
           }
           ret.push_back(cur_level_value);
       }
       return ret;
    }

    // Binary Tree Level Order Traversal II
    vector<vector<int> > levelOrderBottom(TreeNode *root) {
       vector<vector<int> > ret;
       if (root == NULL)
           return ret;
       vector<TreeNode *> next_level_node;
       vector<TreeNode *> cur_level_node;
       next_level_node.push_back(root);
       while (!next_level_node.empty()) {
           vector<int> cur_level_value;
           cur_level_node = next_level_node;
           next_level_node.clear();
           for (int i = 0; i < cur_level_node.size(); ++i) {
               cur_level_value.push_back(cur_level_node[i]->val);
               if (cur_level_node[i]->left != NULL)
                    next_level_node.push_back(cur_level_node[i]->left);
               if (cur_level_node[i]->right != NULL)
                    next_level_node.push_back(cur_level_node[i]->right);
           }
           ret.push_back(cur_level_value);
       }
       std::reverse(ret.begin(), ret.end());
       return ret;
    }

    // Binary Tree Zigzag Level Order Traversal
    vector<vector<int> > zigzagLevelOrder(TreeNode *root) {
       vector<vector<int> > ret;
       if (root == NULL)
           return ret;
       vector<TreeNode *> next_level_node;
       vector<TreeNode *> cur_level_node;
       next_level_node.push_back(root);
       while (!next_level_node.empty()) {
           vector<int> cur_level_value;
           cur_level_node = next_level_node;
           next_level_node.clear();
           for (int i = 0; i < cur_level_node.size(); ++i) {
               cur_level_value.push_back(cur_level_node[i]->val);
               if (cur_level_node[i]->left != NULL)
                    next_level_node.push_back(cur_level_node[i]->left);
               if (cur_level_node[i]->right != NULL)
                    next_level_node.push_back(cur_level_node[i]->right);
           }
           if (ret.size() % 2)
               std::reverse(cur_level_value.begin(), cur_level_value.end());
           ret.push_back(cur_level_value);
       }
       return ret;
    }

    // Spiral Matrix
    vector<int> spiralOrder(vector<vector<int> > &matrix) {
        vector<int> ret;
        int row_num = matrix.size();
        if (row_num == 0)
            return ret;
        int col_num = matrix[0].size();
        bool visited[row_num][col_num];
        memset(visited, 0, sizeof(bool) * row_num * col_num);
        const int dx[] = {0, 1, 0, -1};
        const int dy[] = {1, 0, -1, 0};
        int x = 0, y = 0;
        int direction = 0;
        for (int i = 0; i < row_num * col_num; ++i) {
            visited[x][y] = true;
            ret.push_back(matrix[x][y]);
            int nx = x + dx[direction];
            int ny = y + dy[direction];
            if (nx < 0 || nx >= row_num || ny < 0 || ny >= col_num || visited[nx][ny]) {
                direction = (direction + 1) % 4;
                nx = x + dx[direction];
                ny = y + dy[direction];
            }
            x = nx;
            y = ny;
        }
        return ret;
    }


    // Spiral Matrix II
    vector<vector<int> > generateMatrix(int n) {
        vector<vector<int> > ret;
        if (n == 0)
            return ret;

        bool visited[n][n];
        memset(visited, 0, sizeof(bool) * n * n);
        int map[n][n];
        const int dx[] = {0, 1, 0, -1};
        const int dy[] = {1, 0, -1, 0};
        int x = 0, y = 0;
        int direction = 0;
        for (int i = 1; i <= n * n; ++i) {
            visited[x][y] = true;
            map[x][y] = i;
            int nx = x + dx[direction];
            int ny = y + dy[direction];
            if (nx < 0 || nx >= n || ny < 0 || ny >= n || visited[nx][ny]) {
                direction = (direction + 1) % 4;
                nx = x + dx[direction];
                ny = y + dy[direction];
            }
            x = nx;
            y = ny;
        }
        for (int i = 0; i < n; ++i) {
            vector<int> row;
            for (int j = 0; j < n; ++j)
                row.push_back(map[i][j]);
            ret.push_back(row);
        }
        return ret;
    }

    string simplifyPath(string path) {
        istringstream ss(path);
        vector<string> paths;
        char tmp[512];
        while (ss.get(tmp, 510, '/')) {
            string p = string(tmp);
            if (p == "." || p.empty())
                continue;
            else if (p == ".." && paths.size() > 0)
                paths.pop_back();
            else
                paths.push_back(p);
        }
        if (paths.size() == 0)
            return "/";
        string ret;
//        for (string x : paths)
//            ret += x + "/";
        return ret;
    }


    // Set Matrix Zeroes
    void setZeroes(vector<vector<int> > &matrix) {
         bool zeroCol = false, zeroRow = false;
         int rows = matrix.size();
         int cols = matrix[0].size();
         for (int i = 0; i < cols; i++) {
             if (matrix[0][i] == 0) {
                 zeroRow = true;
                 break;
             }
         }

         for (int i = 0; i < rows; i++) {
             if (matrix[i][0] == 0) {
                 zeroCol = true;
                 break;
             }
         }

         for (int i = 1; i < rows; i++) {
             for (int j = 1; j < cols; j++) {
                 if (matrix[i][j] == 0)
                     matrix[i][0] = 0;
                     matrix[0][j] = 0;
             }
         }

         for (int i = 1; i < cols; i++) {
             if (matrix[0][i] == 0) {
                 for (int j = 1; j < rows; j++)
                     matrix[j][i] = 0;
             }
         }

         for (int i = 1; i < rows; i++) {
             if (matrix[i][0] == 0) {
                 for (int j = 1; j < cols; j++)
                     matrix[i][j] = 0;
             }
         }

         if (zeroRow) {
             for (int i = 0; i < cols; i++)
                 matrix[0][i] = 0;
         }

         if (zeroCol) {
             for (int i = 0; i < rows; i++)
                 matrix[i][0] = 0;
         }

     }

    int uniquePaths(int m, int n) {
        int matrix[m + 1][n + 1];
        memset(matrix, 0, sizeof(int) * (m + 1) * (n + 1));
        matrix[1][1] = 1;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (i == 1 && j == 1)
                    continue;
                matrix[i][j] = matrix[i - 1][j] + matrix[i][j - 1];
            }
        }
        return matrix[m][n];
    }

    int uniquePathsWithObstacles(vector<vector<int> > &obstacleGrid) {
        int m = obstacleGrid.size();
        if (m == 0)
            return 0;
        int n = obstacleGrid[0].size();
        int matrix[m + 1][n + 1];
        memset(matrix, 0, sizeof(int) * (m + 1) * (n + 1));
        matrix[0][1] = 1;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (obstacleGrid[i - 1][j - 1] == 0)
                    matrix[i][j] = matrix[i - 1][j] + matrix[i][j - 1];
                else
                    matrix[i][j] = 0;
            }
        }
        return matrix[m][n];
    }


    void restoreIpAddressHelper(string &s, int index, int depth,
                                string cur, vector<string>& ret) {
        if (depth == 4) {
            if (index == s.length())
                ret.push_back(cur.substr(0, cur.length() - 1));
            return;
        }
        int tmp = 0;
        string t = "";
        for (int i = index; i < s.length(); i++) {
            tmp = tmp * 10 + (s[i] - '0');
            t += s[i];
            if (tmp <= 255)
                restoreIpAddressHelper(s, i + 1, depth + 1, cur + t + ".", ret);
            else
                break;
            if (tmp == 0)
                break;
        }
    }
    vector<string> restoreIpAddresses(string s) {
        vector<string> ret;
        restoreIpAddressHelper(s, 0, 0, "", ret);
        return ret;
    }

	TreeNode *flattenHelper(TreeNode *root) {
		if (root == NULL)
			return NULL;
		TreeNode *lastLeftNode = NULL, *lastRightNode = NULL;
		if (root->right)
			lastRightNode = flattenHelper(root->right);
		if (root->left) {
			lastLeftNode = flattenHelper(root->left);
			lastLeftNode->right = root->right;
			root->right = root->left;
			root->left = NULL;
		}
		if (lastRightNode)
			return lastRightNode;
		else if (lastLeftNode)
			return lastLeftNode;
		else
			return root;
	}
	void flatten(TreeNode *root) {
		flattenHelper(root);
	}

	vector<vector<int> > subsets(vector<int> &S) {
		vector<vector<int> > ret;
		long long size = pow(2, S.size());
		sort(S.begin(), S.end());
		for (long long i = 0; i < size; ++i) {
			vector<int> t;
			for (int j = 0; j < S.size(); j++) {
				if ((i >> j) % 2)
					t.push_back(S[j]);
				if ((i >> j) == 0)
					break;
			}
			ret.push_back(t);
		}
		return ret;
	}

	// Surrounded Regions
	void solveHelper(vector<vector<char>>& board, int i, int j) {
		if (board[i][j] != 'O')
			return;
		board[i][j] = 'Y';

		if (i - 1 >= 0)
			solveHelper(board, i - 1, j);
		if (i + 1 < board.size())
			solveHelper(board, i + 1, j);
		if (j - 1 >= 0)
			solveHelper(board, i, j - 1);
		if (j + 1 < board[0].size())
			solveHelper(board, i, j + 1);

	}

	void solve(vector<vector<char>> &board) {
		if (board.size() == 0)
			return;

		for (int i = 0; i < board.size(); ++i) {
			solveHelper(board, i, 0);
			solveHelper(board, i, board[0].size() - 1);
		}

		for (int j = 0; j < board[0].size(); ++j) {
			solveHelper(board, 0, j);
			solveHelper(board, board.size() - 1, j);
		}

		for (int i = 0; i < board.size(); ++i) {
			for (int j = 0; j < board[0].size(); ++j) {
				if (board[i][j] == 'O')
					board[i][j] = 'X';
				else if (board[i][j] == 'Y')
					board[i][j] = 'O';
			}
		}
	}

	// Binary Tree Maximum Path Sum
	int maxSum;
	int maxPathSumHelper(TreeNode *root) {
		if (root == NULL)
			return 0;
		maxSum = max(root->val, maxSum);
		int leftMax = INT_MIN, rightMax = INT_MIN;
		int left = INT_MIN, right = INT_MIN;
		if (root->left != NULL) {
			leftMax = maxPathSumHelper(root->left);
			left = leftMax + root->val;
			maxSum = max(left, maxSum);
		}
		if (root->right != NULL) {
			rightMax = maxPathSumHelper(root->right);
			right = rightMax + root->val;
			maxSum = max(right, maxSum);
		}
		if (root->left != NULL && root->right != NULL)
			maxSum = max(leftMax + rightMax + root->val, maxSum);

		return max(max(left, right), root->val);
	}

	int maxPathSum(TreeNode *root) {
		maxSum = INT_MIN;
		if (root == NULL)
			return 0;
		maxSum = max(maxSum, maxPathSumHelper(root));
		return maxSum;
	}


	TreeNode *buildTreeHelper(vector<int> &inorder, vector<int> &postorder,
		int instart, int poststart, int size) {
			if (size <= 0)
				return NULL;
			int val = postorder[postorder + size - 1];
			TreeNode *root = new TreeNode(val);
			int i = instart;
			for (; i < instart + size; i++) {
				if (inorder[i] == val)
					break;
			}
			int leftsize = i - instart;
			root->left = buildTreeHelper(inorder, postorder, instart, poststart, leftsize);
			root->right = buildTreeHelper(inorder, postorder, i + 1, poststart + leftsize, size - leftsize - 1);
			return root;
	}

	TreeNode *buildTree(vector<int> &inorder, vector<int> &postorder) {
		return buildTreeHelper(inorder, postorder, 0, 0, inorder.size());
	}


	TreeNode *buildTreeHelper(vector<int> &preorder, vector<int> &inorder,
		int prestart, int instart, int size) {
			if (size <= 0)
				return NULL;
			int val = preorder[prestart];
			TreeNode *root = new TreeNode(val);
			int i = instart;
			for (; i < instart + size; i++) {
				if (inorder[i] == val)
					break;
			}
			int leftsize = i - instart;
			root->left = buildTreeHelper(preorder, inorder, prestart + 1, instart, leftsize);
			root->right = buildTreeHelper(preorder, inorder, prestart + leftsize + 1, i + 1, size - leftsize - 1);
			return root;
	}

	TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder) {
		return buildTreeHelper(preorder, inorder, 0, 0, inorder.size());
	}


	bool existHelper(vector<vector<char>> &board, vector<vector<bool>>& visited,
		const string& word, int x, int y, int depth, int dx[], int dy[]) {
			if (depth == word.length() - 1)
				return true;

			for (int i = 0; i < 4; i++) {
				int nx = x + dx[i];
				int ny = y + dy[i];
				if (nx >= 0 && nx < board.size()
					&& ny >= 0 && ny < board[0].size()) {
						if (!visited[nx][ny] && board[nx][ny] == word[depth + 1]) {
							visited[nx][ny] = true;
							if (existHelper(board, visited, word, nx, ny, depth + 1, dx, dy))
								return true;
							visited[nx][ny] = false;
						}
				}
			}
			return false;
	}

	bool exist(vector<vector<char> > &board, string word) {
		int rows = board.size();
		int cols = board[0].size();

		int dx[] = {0, -1, 0, 1};
		int dy[] = {-1, 0, 1, 0};

		vector<vector<bool>> visited(rows, vector<bool>(cols, false));

		for (int i = 0; i < board.size(); i++) {
			for (int j = 0; j < board[0].size(); j++) {
				if (board[i][j] == word[0]) {
					visited[i][j] = true;
					if (existHelper(board, visited, word, i, j, 0, dx, dy))
						return true;
					visited[i][j] = false;
				}
			}
		}

		return false;

	}

	ListNode *deleteDuplicates(ListNode *head) {
		if (head == NULL)
			return NULL;
		ListNode *previous = head;
		ListNode *cur = head->next;
		while (cur) {
			if (cur->val != previous->val) {
				previous->next = cur;
				previous = previous->next;
			}
			cur = cur->next;
		}
		previous->next = NULL;
		return head;
	}

	ListNode *deleteDuplicates(ListNode *head) {
		if (head == NULL)
			return head;
		ListNode *ret_head = new ListNode(0);
		ListNode *previous = ret_head;
		ListNode *cur = head;
		ListNode *r = cur->next;
		int count = 1;
		while (r) {
			if (r->val != cur->val) {
				if (count == 1) {
					previous->next = cur;
					previous = previous->next;
				}
				cur = r;
				count = 1;
			} else
				count++;
			r = r->next;
		}
		if (count == 1) {
			previous->next = cur;
			previous = previous->next;
		}
		previous->next = NULL;
		return ret_head->next;
	}

	int divide(int dividend, int divisor) {
		long long a = dividend;
		long long b = divisor;
		int sign = 0;
		if (a < 0) {
			a = -a;
			sign = sign ^ 1;
		}
		if (b < 0) {
			b = -b;
			sign = sign ^ 1;
		}
		long long c = b;
		int offset = 0;
		while (c <= a) {
			c = c << 1;
			offset += 1;
		}
		offset--;
		c = c >> 1;
		long long ret = 0;
		for (int i = offset; i >= 0; i--) {
			if (a >= c && c != 0) {
				ret += 1 << i;
				a -= c;
			}
			c = c >> 1;
		}

		return sign ? -ret : ret;
	}

	string minWindow(string S, string T) {
		int min_start = -1;
		int min_length = INT_MAX;
		vector<int> count_t(256, 0);
		vector<int> count_s(256, 0);

		if (S.length() == 0 || T.length() == 0)
			return "";

		for (int i = 0; i < T.length(); i++) {
			count_t[T[i]]++;
		}

		int count = T.length();
		int start = 0;
		for (int end = 0; end < S.length(); end++) {
			if (count_s[S[end]] < count_t[S[end]]) {
				--count;
			}
			count_s[S[end]]++;
			if (count == 0) {
				while(count_s[S[start]] > count_t[S[start]]) {
					count_s[S[start]]--;
					start++;
				}
				if (end - start + 1< min_length) {
					min_start = start;
					min_length = end - start + 1;
				}
				count_s[S[start]]--;
				count++;
				start++;
			}
		}

		if (min_start < 0)
			return "";
		return S.substr(min_start, min_length);
	}

	void nextPermutation(vector<int> &num) {
		if (num.size() <= 1)
			return;
		int i = num.size() - 2, j;
		while (i >= 0 && num[i] >= num[i + 1])
			--i;
		if (i == -1) {
			sort(num.begin(), num.end());
			return;
		}
		for (j = num.size() - 1; j > i; --j) {
			if (num[j] > num[i])
				break;
		}
		int tmp = num[j];
		num[j] = num[i];
		num[i] = tmp;

		sort(num.begin() + i + 1, num.end());
	}

	void connect(TreeLinkNode *root) {
		if (root == NULL)
			return;
		if (root->left == NULL && root->right == NULL)
			return;
		TreeLinkNode *previousFirst = root;
		while (previousFirst != NULL) {
			TreeLinkNode *previousRunner = previousFirst;
			TreeLinkNode *curFirst = NULL, *curPre = NULL;
			while (previousRunner != NULL) {
				if (previousRunner->left != NULL) {
					if (!curFirst) {
						curFirst = previousRunner->left;
						curPre = curFirst;
					}
					else {
						curPre->next = previousRunner->left;
						curPre = curPre->next;
					}
				}
				if (previousRunner->right != NULL) {
					if (!curFirst) {
						curFirst = previousRunner->right;
						curPre = curFirst;
					}
					else {
						curPre->next = previousRunner->right;
						curPre = curPre->next;
					}

				}
				previousRunner = previousRunner->next;
			}
			if (!curFirst)
				break;
			previousFirst = curFirst;
		}
	}

	vector<vector<char>> bt;

	bool isValidRow(int x, int y) {
		for (int i = 0; i < 9; i++)
			if (i != y && bt[x][i] == bt[x][y])
				return false;
		return true;
	}

	bool isValidCol(int x, int y) {
		for (int i = 0; i < 9; i++)
			if (i != x && bt[i][y] == bt[x][y])
				return false;
		return true;
	}

	bool isValidRect(int x, int y) {
		int tx = x / 3 * 3;
		int ty = y / 3 * 3;
		for (int i = tx; i < tx + 3; ++i) {
			for (int j = ty; j < ty + 3; ++j) {
				if (!(x == i && y == j) && bt[i][j] == bt[x][y])
					return false;
			}
		}
		return true;
	}


	bool solveSudokuHelper() {
		int dx = -1, dy = -1;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if (bt[i][j] == '.') {
					dx = i;
					dy = j;
					break;
				}
			}
		}
		if (dx == -1)
			return true;

		for (int i = 1; i <= 9; i++) {
			bt[dx][dy] = (char)(i + '0');
			if (isValidRow(dx, dy) && isValidCol(dx, dy)
				&& isValidRect(dx, dy) && solveSudokuHelper())
				return true;
			bt[dx][dy] = '.';
		}

		return false;
	}


	void solveSudoku(vector<vector<char> > &board) {
		bt = board;
		solveSudokuHelper();
		board = bt;
	}

	bool isValidSudoku(vector<vector<char> > &board) {
		vector<vector<bool>> col(9, vector<bool>(10, false));
		vector<vector<bool>> row(9, vector<bool>(10, false));
		vector<vector<bool>> rect(9, vector<bool>(10, false));
		for (int i = 0; i < 9; ++i) {
			for (int j = 0; j < 9; ++j) {
				if (board[i][j] == '.')
					continue;
				int num = board[i][j] - '0';
				if (row[i][num])
					return false;
				row[i][num] = true;
				if (col[j][num])
					return false;
				col[j][num] = true;
				int c = i / 3 * 3 + j / 3;
				if (rect[c][num])
					return false;
				rect[c][num] = true;
			}
		}
		return true;
	}

	unordered_map<string, int> strtoid;
	unordered_map<int, string> idtostr;
	vector<vector<int>> maps;
	vector<vector<int>> pre;
	vector<int> distance;
	vector<vector<string>> ret;

	void buildMaps(unordered_set<string> &dict) {
		strtoid.clear();
		idtostr.clear();
		distance.clear();
		maps.clear();
		pre.clear();
		ret.clear();
		unordered_set<string>::iterator it = dict.begin();
		int i = 0;
		while (it != dict.end()) {
			strtoid[*it] = i;
			idtostr[i] = *it;
			maps.push_back(vector<int>());
			pre.push_back(vector<int>());
			distance.push_back(INT_MAX);
			i++;
			++it;
		}
		it = dict.begin();
		while (it != dict.end()) {
			string cur = *it;
			int curId = strtoid[*it];
			for (int i = 0; i < cur.length(); ++i) {
				char origin = cur[i];
				for (int j = 0; j < 26; j++) {
					char newChar = char('a' + j);
					if (newChar == origin)
						continue;
					cur[i] = newChar;
					if (dict.find(cur) != dict.end()) {
						int dstId = strtoid[cur];
						maps[curId].push_back(dstId);
					}
				}
				cur[i] = origin;
			}
			++it;
		}
	}

	void genPath(int startId, int endId, vector<int>& path) {
		if (startId == endId) {
			vector<string> t;
			for (int i = path.size() - 1; i >= 0; --i) 
				t.push_back(idtostr[path[i]]);
			ret.push_back(t);
			return;
		}
		for (int i = 0; i < pre[startId].size(); i++) {
			path.push_back(pre[startId][i]);
			genPath(pre[startId][i], endId, path);
			path.pop_back();
		}
	}
	
    vector<vector<string>> findLadders(string start, string end, unordered_set<string> &dict) {
        dict.insert(start);
		dict.insert(end);
		buildMaps(dict);
		int startId = strtoid[start];
		int endId = strtoid[end];
		queue<int> q;
		q.push(startId);
		distance[startId] = 0;
		while (!q.empty()) {
			int curId = q.front();
			q.pop();
			int d = distance[curId] + 1;
			for (int i = 0; i < maps[curId].size(); ++i) {
				int nextId = maps[curId][i];
				if (distance[nextId] == INT_MAX) {
					q.push(nextId);
					pre[nextId].push_back(curId);
					distance[nextId] = d;
				}
				else if (distance[nextId] == d)
					pre[nextId].push_back(curId);
			}
		}
		vector<int> tmp;
		tmp.push_back(endId);
		genPath(endId, startId, tmp);
		return ret;
    }

	int ladderLength(string start, string end, unordered_set<string> &dict) {
		dict.insert(start);
		dict.insert(end);
		buildMaps(dict);
		int startId = strtoid[start];
		int endId = strtoid[end];
		queue<int> q;
		q.push(startId);
		distance[startId] = 0;
		while (!q.empty()) {
			int curId = q.front();
			if (curId == endId)
				break;
			q.pop();
			int d = distance[curId] + 1;
			for (int i = 0; i < maps[curId].size(); ++i) {
				int nextId = maps[curId][i];
				if (distance[nextId] == INT_MAX) {
					q.push(nextId);
					pre[nextId].push_back(curId);
					distance[nextId] = d;
				}
				else if (distance[nextId] == d)
					pre[nextId].push_back(curId);
			}
		}
		if (distance[endId] == INT_MAX)
			return 0;
		else
			return distance[endId] + 1;
	}

	int minimumTotal(vector<vector<int> > &triangle) {
		int n = triangle.size();
		if (n == 0)
			return 0;
		if (n == 1)
			return triangle[0][0];

		vector<vector<int>> f(2, vector<int>(triangle.size() + 1, INT_MAX));
		for (int i = 0; i < n; ++i)
			f[0][i] = triangle[n - 1][i];
		bool flag = 0;
		for (int i = n - 2; i >= 0; --i) {
			for (int j = i; j >= 0; --j) {
				f[!flag][j] = min(f[flag][j], f[flag][j + 1]) + triangle[i][j];
			}
			flag = !flag;
		}

		return f[flag][0];
	}


	vector<TreeNode *> generateTreesHelper(int start, int end) {
		vector<TreeNode *> ret;
		if (end < start) {
			ret.push_back(NULL);
			return ret;
		}
		if (start == end) {
			ret.push_back(new TreeNode(start));
			return ret;
		}

		for (int i = start; i <= end; i++) {
			vector<TreeNode *> left = generateTreesHelper(start, i - 1);
			vector<TreeNode *> right = generateTreesHelper(i + 1, end);
			for (int j = 0; j < left.size(); j++) {
				for (int k = 0; k < right.size(); k++) {
					TreeNode *root = new TreeNode(i);
					root->left = left[j];
					root->right = right[k];
					ret.push_back(root);
				}
			}

		}
		return ret;
	}
	vector<TreeNode *> generateTrees(int n) {
		return generateTreesHelper(1, n);
	}
};




int main(int argc, char** argv) {

	Solution().divide(10, 3);
	return 0;
}
