# Project Outline: Classifaction of Galaxies and Gravitational Lenses
# General Information
* Wenn ihr Probleme mit Englisch habt, dann können wir auf Deutsch wechseln. In den Tutorien habe ich den Eindruck bekommen, dass Englisch kein Problem darstellt, manche aber Deutsch nicht so gut beherrschen.
* **Almost all communication should be facilitated within one Moodle-forum for both topics**. This way I am able to answer all questions in one place, since I am convinced that
$$
\begin{aligned}
    p(\textrm{Several people have this question} | \textrm{You have this question}) > 0
\end{aligned}
$$
* If your question just requires one line of googling, I will answer your question with a well meant passive aggressive link such as [this](https://letmegooglethat.com/?q=What+is+WSL%3F)
* Emails should only be the communication of choice in very rare cases for very individual and non-content related issues (Sickness, Problems with your Team partners, ...). If you have such a problem, address the email to both Stefan and me.
* It would be nice if you try to help each other. This is also in your own interest, since the more questions you can resolve among yourselves the more time I have left to answer other questions as well as teach and prepare material for you. We should all try to make it such that you get the most out of our time! :)
* If you have written a question in the Moodle-forum and I have not answer it within 36 hours (weekdays), then I have probably overlooked it and you can write me an email.



# Overview
The Weeks here are rough outlines, and of course it depends on your individual learning journey how far you will get, and how much time you will be stuck bug fixing (This happens to the best and you learn a lot, even if it might not feel that way at the time)
Color code: \required{Required}, \optional{Optional}, \advanced{Advanced (Optional)}. If a heading has a certain color assigned to it, it means that all sub headings, and bullet points have the same status unless assigned otherwise.

## Week 1
* Introduction to PyTorch and deep learning
* Setting up the development environment
* Preprocessing the dataset

## Week 2
* Writing Clean Code
* Exploring the dataset
* Building and training a first convolutional neural network (CNN) for image classification
* Evaluating the performance of the CNN


## Week 3
Several possibilities:
* Logging Training Loss with TensorBoard
* Fine-tuning the CNN by adjusting hyperparameters and architecture
* Implementing data augmentation to improve the generalization of the model
* Comparing the performance of the CNN with and without data augmentation
* Visualizing the CNN's decision boundaries and activations

## Week 4
* Investigating advanced techniques for improving the performance of the CNN, such as transfer learning and ensembling
* Creating a presentation to document the project
* Presenting the results and conclusion of the project

# Coding Guidelines

* Use descriptive and meaningful variable names:
    * Choose names that accurately describe the purpose or contents of the variables, rather than using short or ambiguous names
    * Avoid using abbreviations or acronyms that may not be familiar to others reading the code
    * For example, instead of using x and y as variables, use width and height if they represent the dimensions of an object

* Use white space to separate blocks of code and make it more readable:
    * Use blank lines to separate related blocks of code, such as functions or loops
    * Use indentation to show the structure of the code and indicate which lines of code belong to which blocks (this has to be done in python anyway)

* Keep lines of code short and concise, and wrap lines that are too long:
    * If a line of code becomes too long, consider wrapping it and using line continuation characters (such as a backslash) to split it over multiple lines

* Use comments to explain the purpose of code blocks and complex or non-obvious code:
    * Use comments to describe the purpose of code blocks, or to explain complex or non-obvious code

* Use modules and libraries to avoid writing redundant code:
    * Modules are files containing Python code that can be imported and used in other Python code
    * By using modules and libraries, you can avoid having to write code for common tasks or functions, and focus on the specific logic and functionality of your program
    * This way you can also reuse code from other projects or share your own code more easily



## Python docstring
A Python docstring is a string that appears at the beginning of a module, function, class, or method definition, and is used to provide documentation for the code. Good Python docstrings have the following characteristics:

* They are placed immediately after the definition of the module, function, class, or method, on the same line as the definition.
* They are written in triple quotes, so that they can span multiple lines if necessary.
* They provide a brief and concise summary of the code's purpose and behavior.
* They provide a list of the code's arguments and their types or meanings, if applicable.
* They provide a description of the code's return value or output, if applicable.
* They provide examples of how to use the code, if applicable.

Example
```python
def circle_area(radius: float) -> float:
"""
Calculate the area of a circle given its radius.

Parameters:
radius (float): The radius of the circle.

Returns:
float: The area of the circle.

Example:
>>> circle_area(2)
12.566370614359172
"""
return 3.14159 * radius ** 2
```