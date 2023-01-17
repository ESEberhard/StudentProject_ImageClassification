# Week 2

## Code Review:
As we discussed your code will be cross-reviewed. Please send your code to the Group with
$$
\begin{aligned}
    \textrm{ID} = \textrm{YOUR\_OWN\_ID} + 1
\end{aligned}
$$
In a cyclic manner. (Meaning if you are the last group, send it to the first)

### How to review code:
Try to take about 20 minutes to review each others code and focus on the main points, that should be improved.

1. Understand the purpose and context of the code: Before reviewing the code, it's important to understand the context in which the code will be used and what problem it is trying to solve.
2. Check for code organization and readability: Make sure that the code is well-organized and easy to read. This includes proper indentation, meaningful variable and function names, and clear separation of concerns.
3. Check for performance and scalability: Evaluate the code for performance and scalability issues. Look for potential bottlenecks and opportunities for optimization.
4. Check for maintainability: Evaluate the code for maintainability issues. Look for opportunities to refactor the code to make it more modular and easier to update in the future.
5. Be respectful and constructive: Be respectful and constructive when giving feedback on the code. Instead of focusing on what's wrong, focus on what can be improved and how it can be improved.


## Coding Best Practices:

1. Use docstrings to provide documentation for the class and its methods.
2. Use typing for methods to indicate their expected inputs and outputs.
3. Inherit from PyTorch's built-in ```Dataset``` class to take advantage of its functionality.
4. Override the ```__len__``` and ```__getitem__``` methods to define how the data should be accessed.
5. Consider creating a preprocessing method to process data before returning it.
6. Do not use global variables, try to pass as arguments.
7. Use meaningful variable names.
8. Add extra comments where necessary


### Example: Dummy CustomDataset

```python
import torch
from typing import List, Tuple, Any

class CustomDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for loading and processing data.

    Attributes:
        data (list): List of data items.
        labels (list): List of labels for each data item.
    """

    def __init__(self, data: List[Any], labels: List[Any]) -> None:
        """Initializes the dataset with data and labels.

        Args:
            data (List[Any]): List of data items.
            labels (List[Any]): List of labels for each data item.
        """
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns the data item and label at the given index.

        Args:
            idx (int): Index of the data item and label to return.

        Returns:
            Tuple[Any, Any]: Data item and label at the given index.
        """
        return self.data[idx], self.labels[idx]

    def preprocess(self, data: Any) -> Any:
        """Preprocesses the data item.

        Args:
            data (Any): Data item to preprocess.

        Returns:
            Any: Preprocessed data item.
        """
        # Perform any necessary preprocessing here
        return data

# Use the custom dataset in a PyTorch DataLoader
dataloader = torch.utils.data.DataLoader(CustomDataset(data, labels), batch_size=32, shuffle=True)

```